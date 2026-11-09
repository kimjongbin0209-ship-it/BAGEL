"""Run the same seven BAGEL tasks independently on two cat photos."""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import inference


CASES = (
    (
        "01-understand.txt",
        "understand",
        "Describe this image in three concise bullets covering the animal and its coat, "
        "its visible pose or gaze, and the setting. Use only details that are visible.",
    ),
    (
        "02-bushes.png",
        "edit",
        "Replace the background with dense, leafy green bushes. Keep the cat's coat "
        "markings, pose, head direction, position, scale, and camera framing. "
        "Keep the cat naturally supported with realistic contact shadows.",
    ),
    (
        "03-beach.png",
        "edit",
        "Replace the background with a beach, with pale sand beneath the cat and the "
        "ocean in the distance. Keep the cat's coat markings, pose, head direction, "
        "position, scale, and camera framing.",
    ),
    (
        "04-pop-art.png",
        "edit",
        "Restyle this exact scene as bold comic-book pop art with bright contrasting color "
        "blocks, strong ink outlines, and visible halftone dots. Preserve the cat's pose "
        "and the layout of the original scene.",
    ),
    (
        "05-nostalgia.png",
        "edit",
        "Apply a nostalgic warm film-print color grade with gently faded blacks, muted "
        "saturation, and warm highlights. Keep the scene, cat, framing, and fine detail "
        "intact. Change only the color and tone.",
    ),
    (
        "06-grain.png",
        "edit",
        "Add visible fine analog film grain evenly across the image, including smooth "
        "areas. Preserve the existing color balance, brightness, detail, cat, and framing.",
    ),
    (
        "07-breed.png",
        "edit",
        "Change only the cat to a short-haired seal-point Siamese, with a cream body and "
        "dark brown points on the face, ears, paws, and tail wherever visible. Keep its "
        "pose, head direction, expression, position, lighting, framing, and background.",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs=2, type=Path, metavar="PHOTO")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cat-study"))
    parser.add_argument("--model-path", type=Path, default=Path("models/BAGEL-7B-MoT"))
    parser.add_argument("--max-mem-per-gpu", default="40GiB")
    parser.add_argument("--plan-only", action="store_true", help="Write the plan without loading BAGEL")
    args = parser.parse_args()

    inputs = {}
    jobs = []
    settings = [
        "--model-path", str(args.model_path), "--max-mem-per-gpu", args.max_mem_per_gpu,
        "--seed", "42", "--steps", "50", "--max-tokens", "256",
        "--cfg-text-scale", "4", "--cfg-image-scale", "2",
    ]
    for number, photo in enumerate(args.photos, start=1):
        if not photo.is_file():
            parser.error(f"Photo not found: {photo}")
        label = f"cat-{number:02d}"
        inputs[label] = {"file": photo.name, "sha256": sha256(photo.read_bytes()).hexdigest()}
        for filename, task, prompt in CASES:
            destination = args.output_dir / label / filename
            options = ["--task", task, "--image", str(photo), "--prompt", prompt, *settings]
            inference.parse_args([*options, "--output", str(destination)])
            jobs.append((destination, options))

    plan = {
        "inputs": inputs,
        "settings": settings,
        "cases": [{"file": file, "task": task, "prompt": prompt} for file, task, prompt in CASES],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "plan.json"
    if any(destination.exists() for destination, _ in jobs):
        if not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            parser.error("Existing outputs use a different plan. Choose a new --output-dir.")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"Plan: {plan_path} ({len(jobs)} jobs)", flush=True)
    if args.plan_only:
        return

    load_once = lru_cache(maxsize=1)(inference.load_inferencer)
    for destination, options in jobs:
        if destination.is_file():
            print(f"Skipping existing {destination}", flush=True)
            continue
        print(f"Running {destination}", flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as temporary_dir:
            staged_output = Path(temporary_dir) / destination.name
            inference.main([*options, "--output", str(staged_output)], model_loader=load_once)
            staged_output.replace(destination)


if __name__ == "__main__":
    main()
