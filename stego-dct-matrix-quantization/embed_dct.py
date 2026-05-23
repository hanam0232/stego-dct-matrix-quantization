#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image
import math

Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=np.float64)

def text_to_bits(text):
    bits = []
    for ch in text:
        b = format(ord(ch), '08b')
        bits.extend([int(x) for x in b])
    bits.extend([0]*8)
    return bits

def dct2(block):
    N = 8
    result = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            cu = 1/math.sqrt(2) if u == 0 else 1.0
            cv = 1/math.sqrt(2) if v == 0 else 1.0
            s = 0.0
            for x in range(N):
                for y in range(N):
                    s += block[x,y] * math.cos((2*x+1)*u*math.pi/16) * math.cos((2*y+1)*v*math.pi/16)
            result[u,v] = 0.25 * cu * cv * s
    return result

def idct2(block):
    N = 8
    result = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            s = 0.0
            for u in range(N):
                for v in range(N):
                    cu = 1/math.sqrt(2) if u == 0 else 1.0
                    cv = 1/math.sqrt(2) if v == 0 else 1.0
                    s += cu * cv * block[u,v] * math.cos((2*x+1)*u*math.pi/16) * math.cos((2*y+1)*v*math.pi/16)
            result[x,y] = 0.25 * s
    return result

def quantize(dct_block):
    return np.round(dct_block / Q50).astype(int)

def dequantize(q_block):
    return (q_block * Q50).astype(np.float64)

def embed(img_array, bits):
    h, w = img_array.shape
    img_float = img_array.astype(np.float64) - 128.0
    bit_idx = 0
    total_bits = len(bits)
    result = img_float.copy()

    # Lấy và in ma trận khối đầu tiên TRƯỚC khi nhúng
    first_block = img_float[0:8, 0:8].copy()
    C_first_before = quantize(dct2(first_block))
    print("MA TRAN DAU TIEN TRUOC KHI GIAU")
    print(C_first_before)

    for bi in range(0, h - 7, 8):
        if bit_idx >= total_bits:
            break
        for bj in range(0, w - 7, 8):
            if bit_idx >= total_bits:
                break
            block = img_float[bi:bi+8, bj:bj+8].copy()
            D = dct2(block)
            C = quantize(D)
            C[0, 0] = (C[0, 0] & ~1) | bits[bit_idx]
            bit_idx += 1
            D_new = dequantize(C)
            block_new = idct2(D_new)
            result[bi:bi+8, bj:bj+8] = block_new

    # Lấy và in ma trận khối đầu tiên SAU khi nhúng
    first_block_after = result[0:8, 0:8].copy()
    C_first_after = quantize(dct2(first_block_after))
    print("\nMA TRAN DAU TIEN SAU KHI GIAU")
    print(C_first_after)

    result = result + 128.0
    return np.clip(result, 0, 255).astype(np.uint8)

def main():
    if len(sys.argv) != 4:
        sys.exit(1)
    img_path = sys.argv[1]
    message  = sys.argv[2]
    out_path = sys.argv[3]
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    bits = text_to_bits(message)
    result = embed(img_array, bits)
    out_img = Image.fromarray(result)
    out_img.save(out_path)

if __name__ == '__main__':
    main()
