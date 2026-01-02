import os, shutil, glob

SRC = r"data\lfw\lfw_funneled"
DST = r"data\lfw_small\images"
PEOPLE = 100
PER_PERSON = 3

os.makedirs(DST, exist_ok=True)

people = [p for p in sorted(os.listdir(SRC)) if os.path.isdir(os.path.join(SRC,p))]
people = people[:PEOPLE]

copied = 0
kept_people = 0
for person in people:
    src_dir = os.path.join(SRC, person)
    imgs = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))[:PER_PERSON]
    if len(imgs) < 2:
        continue
    out_dir = os.path.join(DST, person)
    os.makedirs(out_dir, exist_ok=True)
    for img in imgs:
        shutil.copy2(img, os.path.join(out_dir, os.path.basename(img)))
        copied += 1
    kept_people += 1

print("People kept:", kept_people)
print("Images copied:", copied)
print("Out:", DST)
