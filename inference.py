"""Command-line inference using the setup from inference.ipynb (CUDA required)."""

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["t2i", "edit", "understand"], default="t2i")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path, help="Input image on this machine")
    parser.add_argument("--model-path", type=Path, default=Path("models/BAGEL-7B-MoT"))
    parser.add_argument("--output", type=Path, help="PNG for t2i/edit; UTF-8 text for understand")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-image-scale", type=float)
    parser.add_argument("--max-mem-per-gpu", default="40GiB", help="Model placement budget, not total VRAM usage")
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.task in ("edit", "understand") and args.image is None:
        parser.error(f"--task {args.task} requires --image")
    if args.task == "t2i" and args.image is not None:
        parser.error("Use --task edit or understand with --image")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Input image not found: {args.image}")
    if args.steps < 1 or args.max_tokens < 1:
        parser.error("--steps and --max-tokens must be positive")
    if args.output is None:
        args.output = Path("results/answer.txt" if args.task == "understand" else "results/output.png")
    if args.image is not None and args.output.resolve() == args.image.resolve():
        parser.error("--output must differ from the input image")
    return args


def load_inferencer(model_path, max_mem_per_gpu):
    import torch
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM,
        SiglipVisionConfig, SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True, visual_und=True, llm_config=llm_config,
        vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
        latent_patch_size=2, max_latent_size=64,
    )
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    device_map = infer_auto_device_map(
        model, max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        dtype=torch.bfloat16,
    )
    same_device_modules = [
        "language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
        "vae2llm", "llm2vae", "connector", "vit_pos_embed",
    ]
    # Resolve a module's device even when Accelerate maps one of its ancestors.
    name = same_device_modules[0]
    while name not in device_map and name:
        name = name.rpartition(".")[0]
    first_device = device_map[name]
    for name in same_device_modules:
        if name in device_map:
            device_map[name] = first_device
    print(f"Device map: {device_map}", flush=True)
    model = load_checkpoint_and_dispatch(
        model, checkpoint=str(model_path / "ema.safetensors"),
        device_map=device_map, offload_buffers=True, dtype=torch.bfloat16,
        force_hooks=True, offload_folder="offload",
    ).eval()
    return InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14), new_token_ids=new_token_ids,
    )


def inference_options(args):
    options = dict(think=args.think, max_think_token_n=args.max_tokens, do_sample=False)
    if args.task == "understand":
        return dict(options, understanding_output=True)
    editing = args.task == "edit"
    return dict(
        options, num_timesteps=args.steps, cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_image_scale if args.cfg_image_scale is not None else (2.0 if editing else 1.0),
        cfg_interval=[0.0 if editing else 0.4, 1.0], timestep_shift=3.0,
        cfg_renorm_min=0.0, cfg_renorm_type="text_channel" if editing else "global",
    )


def main(argv=None, *, model_loader=load_inferencer):
    args = parse_args(argv)
    required = ["llm_config.json", "vit_config.json", "ae.safetensors", "ema.safetensors"]
    missing = [name for name in required if not (args.model_path / name).is_file()]
    if missing:
        raise SystemExit(f"Missing model files in {args.model_path}: {', '.join(missing)}")

    import random
    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise SystemExit("This entry point requires a CUDA GPU; run it on the GPU server.")
    input_image = None
    if args.image is not None:
        from data.data_utils import pil_img2rgb
        with Image.open(args.image) as image:
            input_image = pil_img2rgb(image).copy()
    inferencer = model_loader(args.model_path, args.max_mem_per_gpu)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    result = inferencer(text=args.prompt, image=input_image, **inference_options(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "understand":
        if result["text"] is None:
            raise RuntimeError("Model returned no text")
        args.output.write_text(result["text"], encoding="utf-8")
    else:
        if result["image"] is None:
            raise RuntimeError("Model returned no image")
        result["image"].save(args.output, format="PNG")
    if result.get("text"):
        print(result["text"])
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
