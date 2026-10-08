"""
TimescaleDB Compressed Chunk Decoder for Python.
Decodes hypertable chunks compressed with Delta-of-Delta (DeltaDelta) + Simple8b + RLE.
Supports integer, timestamp, and double precision telemetry metrics.
"""

import base64
import struct
from typing import List, Tuple, Optional

SIMPLE8B_NUM_ELEMENTS = [0, 64, 32, 21, 16, 12, 10, 9, 8, 6, 5, 4, 3, 2, 1]
SIMPLE8B_BIT_LENGTH   = [0,  1,  2,  3,  4,  5,  6, 7, 8, 10, 12, 16, 21, 32, 64, 36]
SIMPLE8B_RLE_SELECTOR = 15


def zigzag_decode(n: int) -> int:
    """Decodes standard zigzag encoded integer."""
    return (n >> 1) ^ (-(n & 1))


def decode_simple8b_stream(buf: bytes, offset: int) -> Tuple[List[int], int]:
    """
    Decodes a Simple8bRleSerialized structure from binary buffer.
    Returns (decoded_values, next_offset).
    """
    num_elem, num_blocks = struct.unpack('>II', buf[offset:offset+8])
    offset += 8

    num_selector_slots = (num_blocks + 15) // 16
    selector_slots = [
        struct.unpack('>Q', buf[offset + i*8 : offset + (i+1)*8])[0]
        for i in range(num_selector_slots)
    ]
    offset += num_selector_slots * 8

    blocks = [
        struct.unpack('>Q', buf[offset + i*8 : offset + (i+1)*8])[0]
        for i in range(num_blocks)
    ]
    offset += num_blocks * 8

    # Extract 4-bit selectors
    selectors = []
    for slot in selector_slots:
        for i in range(16):
            selectors.append((slot >> (4 * i)) & 0xf)
    selectors = selectors[:num_blocks]

    # Decode elements from blocks
    vals = []
    for sel, blk in zip(selectors, blocks):
        if sel == SIMPLE8B_RLE_SELECTOR:
            repeat_count = (blk >> 36) & ((1 << 28) - 1)
            val = blk & ((1 << 36) - 1)
            vals.extend([val] * repeat_count)
        else:
            num_items = SIMPLE8B_NUM_ELEMENTS[sel]
            bit_len = SIMPLE8B_BIT_LENGTH[sel]
            mask = (1 << bit_len) - 1
            for pos in range(num_items):
                val = (blk >> (pos * bit_len)) & mask
                vals.append(val)

    return vals[:num_elem], offset


def decompress_deltadelta_series(b64_str: str) -> List[Optional[int]]:
    """
    Decompresses a base64-encoded TimescaleDB Delta-of-Delta chunk.
    Works for both timestamps (ts_compressed_base64) and values (value_compressed_base64).
    Returns a list of values (or None for nulls).
    """
    buf = base64.b64decode(b64_str)
    if len(buf) < 26:
        return []

    algo = buf[0]
    has_nulls = buf[1]
    last_val, last_delta = struct.unpack('>qq', buf[2:18])
    
    # 1. Decode delta-deltas
    delta_deltas_raw, next_offset = decode_simple8b_stream(buf, 18)
    
    # 2. Decode nulls bitmap if present
    nulls = None
    if has_nulls:
        nulls, _ = decode_simple8b_stream(buf, next_offset)

    # 3. Integrate delta-of-deltas to recover values
    decoded_values: List[Optional[int]] = []
    prev_val = 0
    prev_delta = 0
    raw_idx = 0

    total_elements = len(nulls) if has_nulls and nulls is not None else len(delta_deltas_raw)

    for i in range(total_elements):
        if has_nulls and nulls is not None and i < len(nulls) and nulls[i] == 1:
            decoded_values.append(None)
        else:
            if raw_idx < len(delta_deltas_raw):
                dd = zigzag_decode(delta_deltas_raw[raw_idx])
                raw_idx += 1
                prev_delta += dd
                prev_val += prev_delta
                decoded_values.append(prev_val)
            else:
                decoded_values.append(None)

    return decoded_values
