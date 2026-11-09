"""Measure RGB changes by region and build a hard-mask winter composite."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


IMAGE_SIZE = (768, 768)
BOUNDARY_RADIUS = 5
COMPOSITE_SOURCE = "focused-seed-42-cfg-2.png"
EDIT_FILES = [
    "01-broad-winter.png",
    "02-no-change.png",
    *(
        f"focused-seed-{seed}-cfg-{scale}.png"
        for seed in (42, 43, 44)
        for scale in ("1", "1.5", "2")
    ),
]


def read_image(path, mode):
    with Image.open(path) as image:
        if image.size != IMAGE_SIZE:
            raise ValueError(f"{path} has size {image.size}, expected {IMAGE_SIZE}")
        return image.convert(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("results/winter-study"))
    study_dir = parser.parse_args().study_dir

    source = np.asarray(read_image(study_dir / "inputs/fixture.png", "RGB"))
    mask_path = study_dir / "inputs/mask.png"
    mask = read_image(mask_path, "L")
    mask_values = np.asarray(mask)
    if not np.array_equal(np.unique(mask_values), [0, 255]):
        raise ValueError(f"{mask_path} must contain black background and white foreground")
    foreground = mask_values == 255

    # A boundary pixel has an opposite-class pixel within a 5-pixel square radius.
    filter_width = 2 * BOUNDARY_RADIUS + 1
    foreground_interior = np.asarray(mask.filter(ImageFilter.MinFilter(filter_width))) == 255
    background_interior = np.asarray(mask.filter(ImageFilter.MaxFilter(filter_width))) == 0
    regions = {
        "foreground_interior_mae": foreground_interior,
        "background_interior_mae": background_interior,
        "boundary_band_mae": ~(foreground_interior | background_interior),
    }
    if any(not region.any() for region in regions.values()):
        raise ValueError(f"{mask_path} must have pixels in each measurement region")

    edits = {name: np.asarray(read_image(study_dir / name, "RGB")) for name in EDIT_FILES}
    composite_path = study_dir / "hard-mask-composite.png"
    selected_edit = edits[COMPOSITE_SOURCE]
    Image.fromarray(np.where(foreground[..., None], source, selected_edit)).save(composite_path)
    saved_composite = np.asarray(read_image(composite_path, "RGB"))
    if not np.array_equal(saved_composite[foreground], source[foreground]):
        raise RuntimeError("The saved composite changed a masked source pixel")
    if not np.array_equal(saved_composite[~foreground], selected_edit[~foreground]):
        raise RuntimeError("The saved composite changed an unmasked edit pixel")
    edits[composite_path.name] = saved_composite

    csv_path = study_dir / "metrics.csv"
    source_rgb = source.astype(np.int16)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file", *regions])
        for name, edit in edits.items():
            difference = np.abs(edit.astype(np.int16) - source_rgb)
            writer.writerow([name, *(f"{difference[region].mean():.3f}" for region in regions.values())])

    print(f"Saved RGB mean absolute differences (0–255) to {csv_path}")
    print(f"Verified all {foreground.sum():,} foreground pixels match the source in {composite_path}")


if __name__ == "__main__":
    main()
