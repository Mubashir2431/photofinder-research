from pathlib import Path
import cv2
import numpy as np
from photofinder.models.registry import get_embedder

def cos(x,y):
    return float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))

root = Path(r"data\lfw\lfw_funneled")

# pick a person folder with >=2 images
person_dir = next(d for d in root.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) >= 2)
imgs = sorted(person_dir.glob("*.jpg"))
same1, same2 = imgs[0], imgs[1]

# pick a different person folder with >=1 image
diff_dir = next(d for d in root.iterdir() if d.is_dir() and d != person_dir and len(list(d.glob("*.jpg"))) >= 1)
diff = sorted(diff_dir.glob("*.jpg"))[0]

print("SAME:", same1)
print("SAME:", same2)
print("DIFF:", diff)

imgA = cv2.imread(str(same1))
imgB = cv2.imread(str(same2))
imgC = cv2.imread(str(diff))

preprocs = [
    "insightface","bgr_insightface","rgb_127","bgr_127",
    "rgb_01","bgr_01","rgb_01_m05_s05","bgr_01_m05_s05"
]

for pre in preprocs:
    e = get_embedder("arcface_r100_onnx", arcface_preproc=pre)
    EA = e.embed(imgA)
    EB = e.embed(imgB)
    EC = e.embed(imgC)

    if not EA or not EB or not EC:
        print(pre, "=> detection failed on one of the images")
        continue

    a = EA[0].embedding
    b = EB[0].embedding
    c = EC[0].embedding

    print(pre, "same=", cos(a,b), "diff=", cos(a,c), "dim=", a.shape)
