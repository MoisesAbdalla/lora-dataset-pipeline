# LoRA dataset pipeline — synthetic character dataset → captions → training

An end-to-end pipeline to train a **character LoRA** from just a handful of reference images:
generate a varied synthetic dataset with **Qwen Image 2.1** in **ComfyUI**, caption it, and
train on **Z-Image Turbo** with **AI-Toolkit**. The example in `examples/` uses a fictional 3D fox
mascot; the pipeline is subject-agnostic.

```
reference images (2–6)
      │  workflows/qwen-image-2.1-dataset-6refs.json  +  queue_prompts.py
      ▼
synthetic dataset (25 shots: close-up / waist-up / full body)
      │  manual curation — keep the best variation of each shot
      ▼
captions (make_captions.py)
      │  train_lora.yaml  (AI-Toolkit, Z-Image Turbo + training adapter)
      ▼
checkpoints every 250 steps → pick the best by eye
```

## Files

| File | What it does |
|---|---|
| `workflows/qwen-image-2.1-dataset-6refs.json` | ComfyUI workflow: Qwen Image 2.1 (int8) with up to 6 reference images feeding the text encoder, resolution selector and organized Set/Get nodes. Model download links are in the note node. |
| `queue_prompts.py` | Drives ComfyUI through its HTTP API: builds the workflow in API format, injects references and one prompt per job, randomizes seeds, names outputs by shot number. Can reuse the references of the last UI run (`--from-history`). |
| `examples/prompts.md` | The shot list: a fixed identity prefix, a fixed negative prompt and 25 shots balanced across framings, angles, expressions, outfits and lighting. |
| `make_captions.py` | Writes one `.txt` per image from a JSON map, prefixed with the trigger word, and reports images without captions (and captions whose image was curated out). |
| `train_lora.yaml` | AI-Toolkit config for Z-Image Turbo: LoRA rank 16, multi-resolution buckets, 8-bit quantization for a 24 GB GPU, sample prompts every 250 steps. |

## What I learned building it

- **Close-ups teach identity; the mix teaches generalization.** ~1/3 close-ups, 1/3 waist-up, 1/3
  full body, with varied angles and light — otherwise the LoRA only works in one framing.
- **Use face-only references.** With full-body references the model copies their pose and
  framing, and sometimes produces a collage of the references.
- **Don't describe the body in close-up prompts.** It pulled close-ups toward full-body framing,
  so items tagged `[close-up]` drop that sentence from the prefix automatically.
- **Caption what changes, never the identity.** Outfit, pose, framing, setting and light go in the
  caption; face and colors are left to the trigger word so the LoRA learns them.
- **Check every caption against its image.** Generation doesn't always follow the prompt, and a
  caption describing what the prompt *asked for* instead of what the image *shows* teaches the wrong thing.
- **Generate 2–3 variations per shot and curate.** Curation is where dataset quality comes from.
- **Throttle the GPU.** A full queue running non-stop rebooted my PC; the script now runs one job
  at a time with a cooldown.
- **Save checkpoints often and choose by eye.** The last step is rarely the best one; sample
  prompts every 250 steps show where identity is strong but the model hasn't overfit yet.

## Running

```bash
# 1) ComfyUI running locally with the models from the workflow note; references in ComfyUI/input
python queue_prompts.py --prompts examples/prompts.md --name fox --refs fox_ref1.png fox_ref2.png

# 2) curate ComfyUI/output/lora_dataset into dataset/, then caption it
python make_captions.py --dataset dataset/ --captions examples/captions.json --trigger fxcharacter

# 3) train on a GPU machine with AI-Toolkit
python run.py train_lora.yaml
```

**Stack:** ComfyUI · Qwen Image 2.1 · Z-Image Turbo · AI-Toolkit · Python
