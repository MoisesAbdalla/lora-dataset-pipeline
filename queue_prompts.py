"""Queue a LoRA dataset shot list into ComfyUI (Qwen Image 2.1 edit with reference images).

Builds the workflow in API format, injects the reference images and one prompt per job, and
saves every result under a common prefix so the images land together in ComfyUI/output.

Usage (with ComfyUI running at 127.0.0.1:8188):
    python queue_prompts.py --prompts examples/prompts.md --name fox --refs ref1.png ref2.png
    python queue_prompts.py --prompts examples/prompts.md --name fox --from-history

Light mode (default) sends one job at a time, waits for it to finish and pauses --cooldown
seconds so the GPU can cool down. A full queue running non-stop made my PC reboot.
"""
import argparse
import copy
import json
import random
import re
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8188"


def get(path):
    with urllib.request.urlopen(URL + path, timeout=10) as r:
        return json.load(r)


def post(path, data):
    req = urllib.request.Request(URL + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def wait_done(prompt_id):
    while True:
        try:
            h = get(f"/history/{prompt_id}")
        except OSError:
            h = {}
        if prompt_id in h:
            st = h[prompt_id]["status"].get("status_str")
            files = [i["filename"] for o in h[prompt_id]["outputs"].values() for i in o.get("images", [])]
            return st, files
        time.sleep(5)


def load_prompts(path):
    """Shot list format: a '> ' line after 'Prefix', a '> ' line after 'Negative', then numbered items.

    Items tagged [close-up] drop the 'Body:' sentence from the prefix: describing the full body
    pulled close-up shots toward full-body framing.
    """
    text = Path(path).read_text(encoding="utf-8")
    prefix = re.search(r"Prefix.*?\n> (.+)", text).group(1).strip()
    negative = re.search(r"Negative.*?\n> (.+)", text).group(1).strip()
    items = re.findall(r"^(\d+)\. (.+)$", text, re.M)
    return prefix, negative, [(int(n), p.strip()) for n, p in items]


def build_base(args, refs):
    """API-format workflow (equivalent to workflows/*.json with Set/Get nodes resolved)."""
    wf = {
        "477": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "478": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "479": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "484": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "int8", "model": ["477", 0]}},
        "13": {"class_type": "ResolutionSelector", "inputs": {"aspect_ratio": args.aspect, "megapixels": 1.0, "multiple": 32}},
        "480": {"class_type": "EmptyLatentImage", "inputs": {"width": ["13", 0], "height": ["13", 1], "batch_size": 1}},
        "485": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": "", "negative_prompt": "", "resolution": args.ref_res, "clip": ["478", 0], "vae": ["479", 0]}},
        "482": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": args.steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                                                      "model": ["484", 0], "positive": ["485", 0], "negative": ["485", 1], "latent_image": ["480", 0]}},
        "481": {"class_type": "VAEDecode", "inputs": {"samples": ["482", 0], "vae": ["479", 0]}},
        "461": {"class_type": "SaveImageAdvanced", "inputs": {"filename_prefix": "lora_dataset/x", "format": "png", "format.bit_depth": "8-bit",
                                                               "format.input_color_space": "sRGB", "images": ["481", 0]}},
    }
    for i, name in enumerate(refs, start=1):
        nid = str(469 + i)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["485"]["inputs"][f"images.image_{i}"] = [nid, 0]
    return wf


def refs_from_history():
    """Reference images (image_1..N order) from the last run made through the ComfyUI interface."""
    for item in get("/history?max_items=50").values():
        p = item["prompt"][2]
        enc = next((n for n in p.values() if n["class_type"] == "TextEncodeQwenImage21"), None)
        if not enc or item["prompt"][3].get("comfy_usage_source") != "comfyui-frontend":
            continue
        slots = sorted((int(k.split("_")[-1]), v[0]) for k, v in enc["inputs"].items()
                       if k.startswith("images.") and isinstance(v, list))
        return [p[nid]["inputs"]["image"] for _, nid in slots]
    raise SystemExit("Run the workflow once from the ComfyUI interface before using --from-history.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="shot list file (see examples/prompts.md)")
    ap.add_argument("--name", required=True, help="output filename prefix, e.g. the character name")
    ap.add_argument("--refs", nargs="*", default=[], help="reference images in ComfyUI/input (up to 6)")
    ap.add_argument("--from-history", action="store_true", help="reuse the references of the last UI run")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=999)
    ap.add_argument("--per-prompt", type=int, default=1, help="variations per prompt (keep the best one)")
    ap.add_argument("--aspect", default="3:4 (Portrait Standard)")
    ap.add_argument("--ref-res", type=int, default=512, help="reference resolution in the encoder")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cooldown", type=int, default=60, help="pause (s) between images for the GPU to cool down")
    ap.add_argument("--all-at-once", action="store_true", help="queue everything at once (heavy)")
    args = ap.parse_args()

    refs = refs_from_history() if args.from_history else args.refs
    if not refs:
        raise SystemExit("Pass --refs or --from-history.")
    base = build_base(args, refs[:6])
    prefix, negative, prompts = load_prompts(args.prompts)
    print(f"{len(prompts)} prompts, {len(refs[:6])} references")

    count = 0
    for num, p in prompts:
        if not args.start <= num <= args.end:
            continue
        pre = prefix
        if p.startswith("[close-up]"):
            pre = re.sub(r" Body:.*?\.(?= |$)", "", pre)
        text = p.removeprefix("[close-up]").strip()
        for v in range(args.per_prompt):
            wf = copy.deepcopy(base)
            wf["485"]["inputs"]["prompt"] = f"{pre} {text}"
            wf["485"]["inputs"]["negative_prompt"] = negative
            wf["482"]["inputs"]["seed"] = random.randint(0, 2**48)
            wf["461"]["inputs"]["filename_prefix"] = f"lora_dataset/{args.name}_p{num:02d}"
            r = post("/prompt", {"prompt": wf})
            count += 1
            print(f"#{num:02d} v{v + 1} -> {r.get('prompt_id')}  {text[:60]}", flush=True)
            if not args.all_at_once:
                t0 = time.time()
                st, files = wait_done(r["prompt_id"])
                print(f"    {st} in {time.time() - t0:.0f}s {files}; cooling {args.cooldown}s", flush=True)
                time.sleep(args.cooldown)
    print(f"{count} jobs {'queued' if args.all_at_once else 'done'}.")


if __name__ == "__main__":
    main()
