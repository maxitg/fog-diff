#!/usr/bin/env python3

import math
import sys

import numpy as np
import zlib

MAP_WIDTH = 512
TILE_WIDTH_OFFSET = 7
TILE_WIDTH = 1 << TILE_WIDTH_OFFSET
TILE_HEADER_LEN = TILE_WIDTH ** 2
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

    @classmethod
    def from_file(cls, filename: str):
        raw_data = open(filename, 'rb').read()
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
        with open(filename, 'wb') as f:
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
            if int.from_bytes(diff_block, byteorder='big') == 0:
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
        int_block1 = int.from_bytes(block1, byteorder='big')
        int_block2 = int.from_bytes(block2, byteorder='big')

        # Compute the difference
        diff = int_block2 & ~int_block1

        # Convert the difference back to bytes
        return diff.to_bytes(BLOCK_BITMAP_SIZE, byteorder='big')

    #def __init__(self, filename: str):
    #    if not filename:
    #        return
#
    #    self.raw_data = open(filename, 'rb').read()
    #    self.data = zlib.decompress(self.raw_data)
#
    #    self.header = np.frombuffer(self.data[:TILE_HEADER_SIZE], dtype=np.uint16)
#
    #    self.blocks = {}
    #    self.block_extras = {}
#
    #    # print(f"Length of {filename}: {len(self.header)}")
#
    #    # The block represents a grid of pixels TILE_WIDTH x TILE_WIDTH.
    #    # Print the grid as x's and spaces.
    #    for i in range(len(self.header)):
    #        block_idx = self.header[i]
    #        if block_idx > 0:
    #            block_x = i % TILE_WIDTH
    #            block_y = i // TILE_WIDTH
    #            start_offset = TILE_HEADER_SIZE + (block_idx - 1) * BLOCK_SIZE
    #            end_offset = start_offset + BLOCK_SIZE
    #            block_data = self.data[start_offset:end_offset]
    #            
    #            block_pixels = self.convert_block_data_to_bools(block_data)
#
    #            self.blocks[(block_x, block_y)] = block_pixels
    #            self.block_extras[block_idx] = block_data[BLOCK_BITMAP_SIZE:]
#
    #            # print(f"Block {i}: {block_idx}, x: {block_x}, y: {block_y}")
    #            # print(f"Data length: {len(block_pixels)}")
    #            # for i in range(64):
    #            #     for j in range(64):
    #            #         print('x' if block_pixels[i * 64 + j] else ' ', end='')
    #            #     print()
#
    #@classmethod
    #def diff(cls, tile1, tile2):
    #    

    #def convert_block_data_to_bools(self, block_data):
    #    bools = []
    #    for byte in block_data[:BLOCK_BITMAP_SIZE]:  # Only consider bitmap part
    #        for bit in range(8):
    #            bools.append((byte & (1 << bit)) != 0)
    #    return bools


# Replace these paths with your actual file paths
    


tile_file_1 = sys.argv[1]
tile_file_2 = sys.argv[2]
diff_file = sys.argv[3]

tile1 = Tile.from_file(tile_file_1)
tile2 = Tile.from_file(tile_file_2)
diff = Tile.diff(tile1, tile2)
diff.to_file(diff_file)

#diff_data = diff_tiles(tile_file_1, tile_file_2)
#save_diff(diff_data, diff_file)
