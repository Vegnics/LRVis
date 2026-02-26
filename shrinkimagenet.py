from PIL import Image
import os

root = "/home/quinoa/imagenet/train"
out_root = "/home/quinoa/imagenet/train_small"

bad_files = []

#"""
for dirpath, _, filenames in os.walk(root):
    reldpath = dirpath.split("/")
    #newdpath = os.path.join(out_root,os.path.join(*reldpath[5:]))
    newdpath = os.path.join(out_root,*reldpath[5:])
    os.makedirs(newdpath,exist_ok=True)
    print(newdpath)
    for fname in filenames:
        path = os.path.join(dirpath, fname)
        npath = os.path.join(newdpath, fname)
        try:
            with Image.open(path) as img:
                img.verify()
                imgsrc = Image.open(path)
                try:
                    width, height = imgsrc.size # Get original dimensions
                    new_width = 256
                    new_height = int(height * (new_width / width))
                    resized_img_prop = imgsrc.resize((new_width, new_height), Image.Resampling.NEAREST)
                    resized_img_prop.save(npath)
                except Exception:
                    print(f"Cannot resize: {path}||| {npath}")
        except Exception:
            print(f"Corrupted file: {path}")
            bad_files.append(path)

print("Corrupted:", len(bad_files))

for f in bad_files:
    print("Removing:", f)
    os.remove(f)
#"""
