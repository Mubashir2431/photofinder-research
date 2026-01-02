import os
import re
import cv2
from sklearn.datasets import fetch_lfw_people

OUT_ROOT = r"data\lfw_sklearn\images"

def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", s).strip()
    s = s.replace(" ", "_")
    return s[:80] if s else "unknown"

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # min_faces_per_person=2 keeps only identities that can be evaluated (>=2 images)
    bunch = fetch_lfw_people(min_faces_per_person=2, resize=1.0, color=False, download_if_missing=True)

    images = bunch.images      # (N, H, W), grayscale
    labels = bunch.target      # (N,)
    names = bunch.target_names # (num_people,)

    counts = {}
    for i, (img, lab) in enumerate(zip(images, labels)):
        person = safe_name(names[lab])
        person_dir = os.path.join(OUT_ROOT, person)
        os.makedirs(person_dir, exist_ok=True)

        counts[person] = counts.get(person, 0) + 1
        out_path = os.path.join(person_dir, f"{counts[person]:04d}.jpg")

        # convert float -> uint8 and save
        img_u8 = (img * 255).clip(0, 255).astype("uint8") if img.max() <= 1.0 else img.astype("uint8")
        cv2.imwrite(out_path, img_u8)

    print("Saved dataset to:", OUT_ROOT)
    print("People:", len(names))
    print("Images:", len(images))

if __name__ == "__main__":
    main()
