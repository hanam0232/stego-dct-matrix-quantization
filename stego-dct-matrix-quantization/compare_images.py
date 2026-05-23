#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    sys.exit(1)

path1, path2 = sys.argv[1], sys.argv[2]
img1 = Image.open(path1).convert('L')
img2 = Image.open(path2).convert('L')
a1, a2 = np.array(img1), np.array(img2)
psnr_val = "N/A"

if a1.shape == a2.shape:
    mse = np.mean((a1.astype(float) - a2.astype(float))**2)
    if mse > 0:
        psnr_val = f"{10 * np.log10(255**2 / mse):.2f} dB"
    else:
        psnr_val = "inf"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img1, cmap='gray')
axes[0].set_title("Cover")
axes[0].axis('off')
axes[1].imshow(img2, cmap='gray')
axes[1].set_title("Stego")
axes[1].axis('off')
plt.suptitle(f"PSNR: {psnr_val}")
plt.tight_layout()
plt.show()
