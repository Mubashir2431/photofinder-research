import os
import cv2
import numpy as np

root = r"data\test\images"
people = ["Ali", "Sara"]

for person in people:
    os.makedirs(os.path.join(root, person), exist_ok=True)
    for i in range(1, 4):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:] = (i*50 % 255, 120 if person=="Ali" else 200, 180)
        cv2.putText(img, f"{person}-{i}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        out = os.path.join(root, person, f"{i}.png")
        cv2.imwrite(out, img)

print("Created:", root)
