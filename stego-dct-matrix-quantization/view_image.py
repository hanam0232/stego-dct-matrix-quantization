#!/usr/bin/env python3
import sys
from PIL import Image
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit(1)

img_path = sys.argv[1]
img = Image.open(img_path).convert('L')
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.title(img_path)
plt.axis('off')
plt.tight_layout()
plt.show()
