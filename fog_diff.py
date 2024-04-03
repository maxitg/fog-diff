#!/usr/bin/env python3

import os
import shutil
import sys
import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Constants and file format hints were taken from
# https://github.com/CaviarChen/fog-machine/blob/272528056ea47d95e9d784227634cc8ece32f22e/editor/src/utils/FogMap.ts
TILE_WIDTH_OFFSET = 7
TILE_WIDTH = 1 << TILE_WIDTH_OFFSET
TILE_HEADER_LEN = TILE_WIDTH**2
TILE_HEADER_SIZE = TILE_HEADER_LEN * 2
BLOCK_BITMAP_SIZE = 512
BLOCK_EXTRA_DATA = 3
BLOCK_SIZE = BLOCK_BITMAP_SIZE + BLOCK_EXTRA_DATA


class Tile:
    def __init__(
        self,
        blocks: Dict[Tuple[int, int], bytes],
        block_extras: Dict[Tuple[int, int], bytes],
    ):
        self.blocks = blocks
        self.block_extras = block_extras

        # Make header. The header is a 128x128 grid of 16-bit integers.
        # Each integer represents the index of the block in the tile.
        self.header = np.zeros((TILE_WIDTH, TILE_WIDTH), dtype=np.uint16)
        index = 1
        for (x, y), _ in self.blocks.items():
            self.header[y, x] = index
            index += 1

        # Make data. The data is a concatenation of the header and the blocks + block_extras.
        header_data = self.header.flatten().tobytes()

        self.data = header_data
        for (x, y), block_data in self.blocks.items():
            self.data += block_data + self.block_extras[(x, y)]

        # Make raw data. The raw data is the compressed data.
        self.raw_data = zlib.compress(self.data)

    @property
    def blocks_count(self) -> int:
        return len(self.blocks)

    @classmethod
    def from_file(cls, filename: Path) -> "Tile":
        raw_data = open(filename, "rb").read()
        try:
            data = zlib.decompress(raw_data)
        except zlib.error:
            print(f"Failed to decompress {filename}.")
            raise
        header = np.frombuffer(data[:TILE_HEADER_SIZE], dtype=np.uint16)

        blocks = {}
        block_extras = {}

        for i in range(len(header)):
            block_idx = header[i]
            if block_idx > 0:
                block_x = i % TILE_WIDTH
                block_y = i // TILE_WIDTH
                start_offset = TILE_HEADER_SIZE + (block_idx - 1) * BLOCK_SIZE
                end_offset = start_offset + BLOCK_SIZE
                block_data = data[start_offset:end_offset]

                blocks[(block_x, block_y)] = block_data[:BLOCK_BITMAP_SIZE]
                block_extras[(block_x, block_y)] = block_data[BLOCK_BITMAP_SIZE:]

        return cls(blocks, block_extras)

    def to_file(self, filename: Path) -> None:
        """
        Save the tile to a file.
        """
        with open(filename, "wb") as f:
            f.write(self.raw_data)

    @classmethod
    def diff(cls, tile1: "Tile", tile2: "Tile") -> "Tile":
        """
        Compute the difference between two tiles.
        """
        blocks = {}
        block_extras = {}

        for (x, y), block2 in tile2.blocks.items():
            if (x, y) not in tile1.blocks:
                blocks[(x, y)] = block2
                block_extras[(x, y)] = tile2.block_extras[(x, y)]
                continue
            block1 = tile1.blocks[(x, y)]
            diff_block = cls.diff_blocks(block1, block2)

            # If diff_block is empty, skip it
            if int.from_bytes(diff_block, byteorder="big") == 0:
                continue

            blocks[(x, y)] = diff_block
            block_extras[(x, y)] = tile1.block_extras[(x, y)]

        return cls(blocks, block_extras)

    @staticmethod
    def diff_blocks(block1: bytes, block2: bytes) -> bytes:
        """
        Compute the difference between two blocks.
        """
        # We need a bytes object of the same size, so that only bits that appear in block2 but not in block1 are set.

        # Convert the blocks to integers
        int_block1 = int.from_bytes(block1, byteorder="big")
        int_block2 = int.from_bytes(block2, byteorder="big")

        # Compute the difference
        diff = int_block2 & ~int_block1

        # Convert the difference back to bytes
        return diff.to_bytes(BLOCK_BITMAP_SIZE, byteorder="big")


def diff_files(first: Path, second: Path, output: Path) -> None:
    tile1 = Tile.from_file(first)
    tile2 = Tile.from_file(second)
    diff = Tile.diff(tile1, tile2)
    if diff.blocks_count > 0:
        diff.to_file(output)


def diff_directories(first: Path, second: Path, output: Path) -> None:
    os.makedirs(output)

    for second_path in Path(second).rglob("*"):
        relative_path = second_path.relative_to(second)
        first_path = Path(first) / relative_path
        output_path = Path(output) / relative_path

        if not first_path.exists():
            shutil.copy(second_path, output_path)
            continue

        diff_files(first_path, second_path, output_path)


def main(first: str, second: str, output: str):
    if not Path(first).exists():
        print(f"{first} does not exist.")
        return 1

    if not Path(second).exists():
        print(f"{second} does not exist.")
        return 1

    if Path(output).exists():
        print(f"{output} already exists.")
        return 1

    if Path(first).is_dir() and Path(second).is_dir():
        diff_directories(Path(first), Path(second), Path(output))
        return 0

    if Path(first).is_file() and Path(second).is_file():
        diff_files(Path(first), Path(second), Path(output))
        return 0

    print(f"Both {first} and {second} must be either files or directories.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <first> <second> <output>")
        sys.exit(1)

    sys.exit(main(*sys.argv[1:]))
