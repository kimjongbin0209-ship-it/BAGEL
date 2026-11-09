"""Compare BAGEL winter edits on a single input image."""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import inference


FOCUSED_PROMPT = (
    "Replace only the background with a winter scene with visible falling snow. "
    "Keep snow behind the toy and off the foreground. Preserve the toy's pose, "
    "shape, position, size, stripes, and framing. Keep the STUDIO.07 label "
    "legible and unchanged, with its original text and placement."
)

CASES = (
    ("01-broad-winter.png", "Give this photo a winter look.", 42, 2.0),
    ("02-no-change.png", "Return this image unchanged. Preserve every visible detail.", 42, 2.0),
    *(
        (f"focused-seed-{seed}-cfg-{scale:g}.png", FOCUSED_PROMPT, seed, scale)
        for seed in (42, 43, 44)
        for scale in (1.0, 1.5, 2.0)
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/winter-study"))
    parser.add_argument("--model-path", type=Path, default=Path("models/BAGEL-7B-MoT"))
    parser.add_argument("--max-mem-per-gpu", default="40GiB")
    parser.add_argument("--plan-only", action="store_true", help="Write the plan without loading BAGEL")
    args = parser.parse_args()

    if not args.fixture.is_file():
        parser.error(f"Fixture not found: {args.fixture}")
    plan_path = args.output_dir / "plan.json"
    if plan_path.resolve() == args.fixture.resolve():
        parser.error("The fixture cannot be the plan file.")

    settings = [
        "--task", "edit", "--model-path", str(args.model_path),
        "--max-mem-per-gpu", args.max_mem_per_gpu, "--steps", "50", "--cfg-text-scale", "4",
    ]
    jobs = []
    for filename, prompt, seed, scale in CASES:
        destination = args.output_dir / filename
        options = [
            *settings, "--image", str(args.fixture), "--prompt", prompt,
            "--seed", str(seed), "--cfg-image-scale", str(scale),
        ]
        inference.parse_args([*options, "--output", str(destination)])
        jobs.append((destination, options))

    plan = {
        "input": {"file": args.fixture.name, "sha256": sha256(args.fixture.read_bytes()).hexdigest()},
        "settings": settings,
        "cases": [
            {"file": file, "prompt": prompt, "seed": seed, "cfg_image_scale": scale}
            for file, prompt, seed, scale in CASES
        ],
    }
    # Hash the canonical JSON before adding the digest field itself.
    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    plan["plan_sha256"] = sha256(plan_bytes).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        if any(destination.exists() for destination, _ in jobs) and not plan_path.exists():
            parser.error("Existing outputs have no plan. Choose a new --output-dir.")
        with TemporaryDirectory(dir=args.output_dir) as temporary_dir:
            staged_plan = Path(temporary_dir) / plan_path.name
            staged_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            try:
                plan_path.hardlink_to(staged_plan)
            except FileExistsError:
                pass  # Another invocation published its plan first.
    try:
        saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parser.error("The existing plan is unreadable. Choose a new --output-dir.")
    if saved_plan != plan:
        parser.error("The output directory uses a different plan. Choose a new --output-dir.")

    print(f"Plan: {plan_path} ({len(jobs)} jobs)", flush=True)
    if args.plan_only:
        return

    load_once = lru_cache(maxsize=1)(inference.load_inferencer)
    for destination, options in jobs:
        if destination.is_file():
            print(f"Skipping existing {destination}", flush=True)
            continue
        print(f"Running {destination}", flush=True)
        with TemporaryDirectory(dir=args.output_dir) as temporary_dir:
            staged_output = Path(temporary_dir) / destination.name
            inference.main([*options, "--output", str(staged_output)], model_loader=load_once)
            try:
                destination.hardlink_to(staged_output)
            except FileExistsError:
                if not destination.is_file():
                    raise


if __name__ == "__main__":
    main()
