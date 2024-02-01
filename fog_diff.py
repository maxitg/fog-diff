#!/usr/bin/env python3

import os
import pathlib
import shutil
import sys
import zlib

import numpy as np

MAP_WIDTH = 512
TILE_WIDTH_OFFSET = 7
TILE_WIDTH = 1 << TILE_WIDTH_OFFSET
TILE_HEADER_LEN = TILE_WIDTH**2
TILE_HEADER_SIZE = TILE_HEADER_LEN * 2
BLOCK_BITMAP_SIZE = 512
BLOCK_EXTRA_DATA = 3
BLOCK_SIZE = BLOCK_BITMAP_SIZE + BLOCK_EXTRA_DATA
BITMAP_WIDTH_OFFSET = 6
BITMAP_WIDTH = 1 << BITMAP_WIDTH_OFFSET
ALL_OFFSET = TILE_WIDTH_OFFSET + BITMAP_WIDTH_OFFSET


class Tile:
    def __init__(self, blocks, block_extras):
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
    def blocks_count(self):
        return len(self.blocks)

    @classmethod
    def from_file(cls, filename: str):
        raw_data = open(filename, "rb").read()
        data = zlib.decompress(raw_data)
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

    def to_file(self, filename: str):
        """
        Save the tile to a file.
        """
        with open(filename, "wb") as f:
            f.write(self.raw_data)

    @classmethod
    def diff(cls, tile1, tile2):
        """
        Compute the difference between two tiles.
        """
        blocks = {}
        block_extras = {}

        for (x, y), block1 in tile1.blocks.items():
            block2 = tile2.blocks[(x, y)]
            diff_block = cls.diff_blocks(block1, block2)

            # If diff_block is empty, skip it
            if int.from_bytes(diff_block, byteorder="big") == 0:
                continue

            blocks[(x, y)] = diff_block
            block_extras[(x, y)] = tile1.block_extras[(x, y)]

        return cls(blocks, block_extras)

    @staticmethod
    def diff_blocks(block1, block2):
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


def diff_files(first, second, output):
    tile1 = Tile.from_file(first)
    tile2 = Tile.from_file(second)
    diff = Tile.diff(tile1, tile2)
    if diff.blocks_count > 0:
        diff.to_file(output)


def diff_directories(first, second, output):
    os.makedirs(output)

    for second_path in pathlib.Path(second).rglob("*"):
        relative_path = second_path.relative_to(second)
        first_path = pathlib.Path(first) / relative_path
        output_path = pathlib.Path(output) / relative_path

        if not first_path.exists():
            shutil.copy(second_path, output_path)
            continue

        diff_files(first_path, second_path, output_path)


def main(first, second, output):
    if not pathlib.Path(first).exists():
        print(f"{first} does not exist.")
        return 1

    if not pathlib.Path(second).exists():
        print(f"{second} does not exist.")
        return 1

    if pathlib.Path(output).exists():
        print(f"{output} already exists.")
        return 1

    if pathlib.Path(first).is_dir() and pathlib.Path(second).is_dir():
        diff_directories(first, second, output)
        return 0

    if pathlib.Path(first).is_file() and pathlib.Path(second).is_file():
        diff_files(first, second, output)
        return 0

    print(f"Both {first} and {second} must be either files or directories.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <first> <second> <output>")
        sys.exit(1)

    sys.exit(main(*sys.argv[1:]))
