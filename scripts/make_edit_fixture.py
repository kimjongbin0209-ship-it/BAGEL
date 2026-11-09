"""Draw the original winter edit fixture and its foreground mask without dependencies."""

import math
from pathlib import Path
import random
import struct
import zlib


SIZE = 768
LABEL_GLYPHS = {
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00000", "00100"),
    "0": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
}


def png_chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def write_png(path, pixels, channels):
    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, {1: 0, 3: 2}[channels], 0, 0, 0)
    stride = SIZE * channels
    rows = b"".join(b"\0" + pixels[start : start + stride] for start in range(0, len(pixels), stride))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )


def make_backdrop():
    rng = random.Random(471)
    pixels = bytearray()
    lights = [(126, 172, 34, 12), (633, 132, 46, 17), (625, 350, 25, 11)]
    for y in range(SIZE):
        for x in range(SIZE):
            glow = 15 * math.exp(-((x - 366) / 320) ** 2 - ((y - 324) / 325) ** 2)
            for light_x, light_y, spread, brightness in lights:
                distance_squared = (x - light_x) ** 2 + (y - light_y) ** 2
                glow += brightness * math.exp(-distance_squared / (2 * spread**2))
            grain = rng.choice((-1, 0, 0, 0, 1))
            tone = round(glow) + grain
            pixels.extend((11 + tone, 20 + tone, 38 + tone))
    return pixels


class Fixture:
    def __init__(self):
        self.pixels = make_backdrop()
        self.mask = bytearray(SIZE * SIZE)

    def paint_pixel(self, x, y, color, coverage=1.0):
        index = y * SIZE + x
        self.mask[index] = 255
        start = index * 3
        if coverage == 1.0:
            self.pixels[start : start + 3] = bytes(color)
        else:
            for channel, value in enumerate(color):
                position = start + channel
                self.pixels[position] = round(
                    self.pixels[position] * (1.0 - coverage) + value * coverage
                )

    def rectangle(self, left, top, right, bottom, color):
        for y in range(top, bottom):
            for x in range(left, right):
                self.paint_pixel(x, y, color)

    def ellipse(self, cx, cy, rx, ry, color, stripe_height=0):
        edge_scale = min(rx, ry)
        for y in range(cy - ry - 1, cy + ry + 1):
            for x in range(cx - rx - 1, cx + rx + 1):
                radius = math.hypot((x + 0.5 - cx) / rx, (y + 0.5 - cy) / ry)
                coverage = min(1.0, 0.5 + (1.0 - radius) * edge_scale)
                if coverage > 0:
                    fill = (27, 46, 78) if stripe_height and (y // stripe_height) % 2 else color
                    self.paint_pixel(x, y, fill, coverage)

    def draw_toy(self):
        ink = (17, 30, 52)
        fur = (220, 195, 167)
        light_fur = (239, 214, 185)
        cream = (246, 240, 226)
        blush = (197, 145, 132)

        # Draw the ears, paws, and torso behind the face.
        for center_x in (285, 483):
            self.ellipse(center_x, 258, 54, 70, ink)
            self.ellipse(center_x, 258, 49, 65, fur)
            self.ellipse(center_x, 258, 32, 44, blush)
        for center_x in (265, 503):
            self.ellipse(center_x, 510, 43, 87, ink)
            self.ellipse(center_x, 510, 38, 81, fur)
        for center_x in (328, 440):
            self.ellipse(center_x, 607, 56, 30, ink)
            self.ellipse(center_x, 607, 51, 25, fur)
        self.ellipse(384, 498, 124, 120, ink)
        self.ellipse(384, 498, 118, 113, cream, stripe_height=22)

        self.ellipse(384, 339, 138, 114, ink)
        self.ellipse(384, 339, 132, 108, fur)
        self.ellipse(384, 330, 123, 96, light_fur)
        for center_x in (335, 433):
            self.ellipse(center_x, 328, 12, 15, ink)
            self.ellipse(center_x - 3, 324, 4, 4, cream)
        for center_x in (310, 458):
            self.ellipse(center_x, 363, 18, 10, blush)
        self.ellipse(384, 376, 43, 27, cream)
        self.ellipse(384, 364, 13, 8, ink)
        self.rectangle(382, 367, 386, 380, ink)
        self.ellipse(384, 382, 14, 4, ink)

    def draw_label(self):
        self.rectangle(507, 612, 636, 647, (17, 30, 52))
        self.rectangle(509, 614, 634, 645, (246, 240, 226))
        for index, letter in enumerate("STUDIO.07"):
            for row, glyph_row in enumerate(LABEL_GLYPHS[letter]):
                for column, bit in enumerate(glyph_row):
                    if bit == "1":
                        x = 519 + 12 * index + 2 * column
                        y = 622 + 2 * row
                        self.rectangle(x, y, x + 2, y + 2, (17, 30, 52))


def main():
    output = Path(__file__).resolve().parents[1] / "results" / "winter-study" / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    fixture = Fixture()
    fixture.draw_toy()
    fixture.draw_label()
    write_png(output / "fixture.png", fixture.pixels, channels=3)
    write_png(output / "mask.png", fixture.mask, channels=1)
    print(output)


if __name__ == "__main__":
    main()
