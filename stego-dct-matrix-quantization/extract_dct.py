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

def quantize(dct_block):
    return np.round(dct_block / Q50).astype(int)

def extract(img_array):
    h, w = img_array.shape
    img_float = img_array.astype(np.float64) - 128.0
    bits = []
    chars = []

    for bi in range(0, h - 7, 8):
        for bj in range(0, w - 7, 8):
            block = img_float[bi:bi+8, bj:bj+8]
            D = dct2(block)
            C = quantize(D)
            bit = int(C[0, 0]) & 1
            bits.append(bit)
            if len(bits) % 8 == 0:
                byte_bits = bits[-8:]
                val = int(''.join(str(b) for b in byte_bits), 2)
                if val == 0:
                    return ''.join(chars)
                chars.append(chr(val))
    return ''.join(chars)

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    img_path = sys.argv[1]
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    message = extract(img_array)
    print(f"Trich xuat tin: {message}")

if __name__ == '__main__':
    main()

