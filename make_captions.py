"""Write one .txt caption per image in a dataset folder (format used by AI-Toolkit / kohya).

Captions describe only what CHANGES between images — outfit, pose, framing, setting, light.
The character's identity (face, colors, proportions) is NOT described: it gets absorbed by the
trigger word. Describing identity in every caption teaches the model that those traits come
from the text, and the LoRA stops reproducing them on its own.

Usage:
    python make_captions.py --dataset dataset/ --captions examples/captions.json --trigger "fxcharacter"
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path, help="folder with the curated images")
    ap.add_argument("--captions", required=True, type=Path, help="JSON {image_stem: caption}")
    ap.add_argument("--trigger", required=True, help="rare token that will summon the character")
    args = ap.parse_args()

    captions = json.loads(args.captions.read_text(encoding="utf-8"))
    images = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp") for p in args.dataset.glob(ext))
    written = 0
    for img in images:
        text = captions.get(img.stem)
        if text:
            img.with_suffix(".txt").write_text(f"{args.trigger}, {text}", encoding="utf-8")
            written += 1
    missing = [p.name for p in images if p.stem not in captions]
    unused = sorted(set(captions) - {p.stem for p in images})
    print(f"{written} captions written")
    if missing:
        print(f"images without caption: {missing}")
    if unused:
        print(f"captions without image (removed during curation?): {unused}")


if __name__ == "__main__":
    main()
