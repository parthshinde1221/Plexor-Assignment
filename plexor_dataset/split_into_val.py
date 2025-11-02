import argparse, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# used for listiing images in the directory
def list_images(p):
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS])

def ensure_pair(img, labels_train):
    lbl = labels_train / (img.stem + ".txt")
    return lbl if lbl.exists() else None

# used for copying and moving directories
def copy_or_move(src,dst,move=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    (shutil.move if move else shutil.copy2)(str(src), str(dst))

def write_listfile(paths, root: Path, out_path: Path):
    # write relative paths like: data/images/train/<file>
    rels = []
    for p in paths:
        # normalize to .jpg in the list if actual extension differs
        # but use actual file’s extension to avoid inconsistency
        rel = (p.relative_to(root.parent) if p.is_absolute() else p).as_posix()
        rels.append(rel)
    out_path.write_text("\n".join(rels) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="dataset root containing images/ and labels/")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    ap.add_argument("--names", nargs="*", default=["person"], help="Class names for data.yaml")
    ap.add_argument("--list_prefix", default="data", help="Prefix used in train.txt/val.txt (e.g., 'data')")
    args = ap.parse_args()

    root = args.root.resolve()
    imgs_train = root / "images" / "train"
    lbls_train = root / "labels" / "train"
    imgs_val = root / "images" / "val"
    lbls_val = root / "labels" / "val"

    assert imgs_train.is_dir(), f"Missing {imgs_train}"
    assert lbls_train.is_dir(), f"Missing {lbls_train}"

    images = list_images(imgs_train)
    pairs = []
    for img in images:
        lbl = ensure_pair(img, lbls_train)
        if lbl is None:
            # allow empty labels: create an empty file
            (lbls_train / f"{img.stem}.txt").write_text("", encoding="utf-8")
            lbl = lbls_train / f"{img.stem}.txt"
        pairs.append((img, lbl))

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = int(round(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # Split
    for img, lbl in val_pairs:
        dst_img = imgs_val / img.name
        dst_lbl = lbls_val / lbl.name
        copy_or_move(img, dst_img, args.move)
        copy_or_move(lbl, dst_lbl, args.move)

    # If moving, remaining train set is already correct. If copying, keep originals as train.

    # Building lists for listfiles
    # For paths like:- data/images/train/frame_000000.jpg
    list_prefix = args.list_prefix.strip("/")

    # Train list = images currently in images/train (after move/copy)
    cur_train_imgs = list_images(imgs_train)
    train_list = [Path(list_prefix) / "images" / "train" / p.name for p in cur_train_imgs]

    # Val list = images now in images/val
    cur_val_imgs = list_images(imgs_val)
    val_list = [Path(list_prefix) / "images" / "val" / p.name for p in cur_val_imgs]

    # Write train.txt / val.txt at dataset root
    write_listfile(train_list, root, root / "train.txt")
    write_listfile(val_list, root, root / "val.txt")

    # Write/refresh data.yaml
    names = {i: n for i, n in enumerate(args.names)}
    data_yaml = (
        f"train: { (root / 'images/train').as_posix() }\n"
        f"val: { (root / 'images/val').as_posix() }\n"
        "names:\n" + "\n".join([f"  {i}: {n}" for i, n in names.items()]) + "\n"
    )
    (root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"Done. Train: {len(train_list)} | Val: {len(val_list)}")
    print(f"Wrote: {root/'data.yaml'}, {root/'train.txt'}, {root/'val.txt'}")

if __name__ == "__main__":
    main()
