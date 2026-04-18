# capo external assets

All large files — datasets, pretrained weights, codec weights, reconstructed
image sets — live outside the git repo. After `bash scripts/setup.sh` wires
up symlinks, CEO reads them from the paths below.

This document is the canonical "what do I need to download" list. It is
organised by target location so you can batch-upload matching subfolders
to a cloud drive and share a single link per group.

> **TL;DR sizes**: ~2.7 GB weights (CLIP shared, no duplicates) + ~74 GB raw cls
> + ~22 GB raw ret + ~6 GB raw seg + ~21 GB QP=1 reconstructed cls (`new/`)
> + 199 MB QP=1 reconstructed ret (`coco2014_cloud/`).
> QP=7 variants are **not** required.

---

## 1. Pretrained model weights

### 1.1 OpenAI CLIP  → `$CLIP_CACHE_SRC`  →  symlinked as `capo_cls/clip_cache/`

| File | Size | Source | Notes |
|---|---:|---|---|
| `RN50.pt` | 244 MB | [OpenAI CLIP](https://github.com/openai/CLIP) | edge-side default |
| `RN101.pt` | 278 MB | idem | alternative edge arch |
| `ViT-B-16.pt` | 335 MB | idem | cloud-side default |
| `ViT-L-14.pt` | 890 MB | idem | optional, large cloud |
| **Subtotal** | **~1.7 GB** | | |

`clip.load(download_root=...)` will auto-download these on first use, but
pre-seeding the cache folder saves time and avoids the GitHub → OpenAI
redirect.

**Shared with retrieval**: `setup.sh` auto-creates
`capo_ret/pretrained/clip/` as a symlink pointing at `capo_cls/clip_cache/`,
so the LAVIS loader (which reads from `RET_CLIP_CACHE`) uses the exact same
weights. No duplicate download needed.

### 1.2 enc2dec codec  → `$ENC2DEC_MODELS_SRC`  →  symlinked as `enc2dec/models/`

| File | Size | Required for | Notes |
|---|---:|---|---|
| `model0_qp1.pk` | 85 MB | **ALL** enc2dec pipelines at QP=1 | **required** |
| `model0_qp7.pk` | 290 MB | QP=7 variants only | **not needed** (QP=7 is skipped in this repo) |

### 1.3 LAVIS / retrieval CLIP cache  (auto-symlinked, no separate download)

Retrieval uses LAVIS's own CLIP loader, which expects the weights under
`capo_ret/pretrained/clip/`. After `setup.sh` runs, this is automatically
a symlink to `capo_cls/clip_cache/` (§1.1), so **no separate download is
needed**. The 3 files used by ret (RN50.pt, ViT-B-16.pt, ViT-L-14.pt) are
a subset of the 4 files already in the shared CLIP cache.

If for some reason you need a truly separate copy, set `RET_PRETRAINED_SRC`
in `.env` and `setup.sh` will use that instead.

### 1.4 MaskCLIP segmentation  →  `$SEG_PRETRAIN_SRC`  →  symlinked as `capo_seg/pretrain/`

These are pre-extracted CLIP backbones + class-name text embeddings used by
MaskCLIP's segmentor. Re-generation is also possible via
`capo_seg/tools/maskclip_utils/convert_clip_weights.py` + `prompt_engineering.py`.

| File | Size | Task |
|---|---:|---|
| `RN50_clip_backbone.pth` | ≈ 90 MB | seg RN50 backbone |
| `RN50_clip_weights.pth` | ≈ 10 MB | seg RN50 decode head proj |
| `ViT16_clip_backbone.pth` | ≈ 330 MB | seg ViT-B/16 backbone |
| `ViT16_clip_weights.pth` | ≈ 6 MB | seg ViT-B/16 decode head proj |
| `ade_RN50_clip_text.pth` | ≈ 1 MB | ADE20K text embeddings (RN50) |
| `ade_RN50_npe_clip_text.pth` | ≈ 1 MB | ADE20K NPE variant |
| `ade_ViT16_clip_text.pth` | ≈ 1 MB | ADE20K text embeddings (ViT-B/16) |
| `ade_ViT16_npe_clip_text.pth` | ≈ 1 MB | ADE20K NPE variant |
| `voc_RN50_clip_text.pth` | ≈ 1 MB | VOC text embeddings (RN50) |
| `voc_RN50_npe_clip_text.pth` | ≈ 1 MB | VOC NPE variant |
| `voc_ViT16_clip_text.pth` | ≈ 1 MB | VOC text embeddings (ViT-B/16) |
| `voc_ViT16_npe_clip_text.pth` | ≈ 1 MB | VOC NPE variant |
| **Subtotal** | **~895 MB** | |

---

## 2. Raw datasets

### 2.1 Classification  →  `$CLS_DATA_SRC`  →  symlinked as `capo_cls/dataset/`  →  `datasets/cls/`

Put each dataset under its canonical sub-folder name (the existing
`split_zhou_*.json` files in [capo_cls/data/data_splits](capo_cls/data/data_splits)
assume this layout).

**Fine-grained datasets (with split_zhou JSON):**

| Sub-folder | Images root | Total images | val+test images | Size (raw) | Split JSON |
|---|---|---:|---:|---:|---|
| `caltech-101/` | `101_ObjectCategories/` | 9,144 | 4,114 | 165 MB | `split_zhou_Caltech101.json` |
| `dtd/` | `images/` | 5,640 | 2,820 | 617 MB | `split_zhou_DescribableTextures.json` |
| `eurosat/` | `2750/` | 27,000 | 13,500 | 137 MB | `split_zhou_EuroSAT.json` |
| `food-101/` | `images/` | 101,000 | 50,500 | 5.0 GB | `split_zhou_Food101.json` |
| `oxford_flowers/` | `jpg/` | 8,189 | 4,096 | 348 MB | `split_zhou_OxfordFlowers.json` |
| `oxford_pets/` | `images/` | 7,390 | 4,405 | 850 MB | `split_zhou_OxfordPets.json` |
| `stanford_cars/` | `cars_train/` + `cars_test/` | 16,185 | 9,676 | 1.9 GB | `split_zhou_StanfordCars.json` |
| `sun397/` | `SUN397/` | 23,820 (pre-filtered) | 23,820 | 44 GB | `split_zhou_SUN397.json` |
| `ucf101/` | `UCF-101-midframes/` | 13,320 | 5,681 | 164 MB | `split_zhou_UCF101.json` |

**Datasets without split JSON (original ↔ new/ 1:1):**

| Sub-folder | Images root | Total images | Size (raw) | Filter |
|---|---|---:|---:|---|
| `fgvc_aircraft/` | `images/` | 10,000 | 2.7 GB | `images_variant_test.txt` → 6,666 val+test |
| `imagenet/` | `images/` | 50,000 | 6.4 GB | full set |
| `imagenet-adversarial/` | `imagenet-a/` | 7,500 | 667 MB | full set |
| `imagenet-rendition/` | `imagenet-r/` | 30,000 | 2.1 GB | full set |
| `imagenet-sketch/` | `images/` | 50,889 | 7.8 GB | full set |
| `imagenetv2/` | `imagenetv2-matched-frequency-format-val/` | 10,000 | 1.2 GB | full set |

| | **Subtotal** | | **~74 GB** | |

### 2.2 Retrieval  →  `$RET_DATA_SRC`  →  symlinked as `capo_ret/dataset/`  →  `datasets/ret/`

| Sub-folder | Contents | Images | Size | Required for |
|---|---|---:|---:|---|
| `coco2014/` | `val2014/` (40,504) + `train2014/` + karpathy JSONs | 40,504 + 40,775 | 16 GB | COCO retrieval |
| `flickr30k/` | `flickr30k-images/` + karpathy JSONs | — | 5.5 GB | Flickr retrieval |
| `nocaps/` | images + `nocaps_val_4500_captions.json` | — | 12 MB | zero-shot captioning |
| **Subtotal** | | | **~22 GB** | |

### 2.3 Segmentation  →  `$SEG_DATA_SRC`  →  symlinked as `capo_seg/data/`  →  `datasets/seg/`

| Sub-folder | Contents | Images | Size | Required for |
|---|---|---:|---:|---|
| `ade/ADEChallengeData2016/` | `images/{training,validation}/`, `annotations/` | ~22K | ~4 GB | ADE20K |
| `VOCdevkit/VOC2012/` | `JPEGImages/`, `SegmentationClass*`, `ImageSets/` | 23,408 | 2.1 GB | VOC 2012 (with aug) |
| **Subtotal** | | | **~6 GB** | |

Cityscapes / PascalContext / COCO-Stuff are supported by configs in
`capo_seg/configs/` but not required by any CEO script. Skip unless needed.

---

## 3. enc2dec-processed datasets (QP=1)

These are the cloud-side reconstructed images. They are either
  (a) **regenerated locally** by running the one-shot preprocessing script, or
  (b) **downloaded pre-built** from your cloud drive (if you uploaded after (a)).

Option (a):

```bash
conda activate joint3
source scripts/_paths.sh
bash scripts/preprocess_enc2dec.sh all         # cls + ret in sequence
# or, in parts:
bash scripts/preprocess_enc2dec.sh cls cuda:0
bash scripts/preprocess_enc2dec.sh ret 4 "0,1,2,3"
```

Option (b) — directory layout to download into place:

### 3.1 cls reconstructions  (inside each `$CLS_DATA_SRC/<dataset>/`)

**Fine-grained (split_zhou JSON, val+test split):**

| Dataset | `new/` images | `new/` size | Filter |
|---|---:|---:|---|
| caltech-101 | 4,114 | 126 MB | `split_zhou_Caltech101.json` → test |
| dtd | 2,820 | 220 MB | `split_zhou_DescribableTextures.json` → test |
| eurosat | 13,500 | 107 MB | `split_zhou_EuroSAT.json` → test |
| food-101 | 50,500 | 2.7 GB | `split_zhou_Food101.json` → test |
| oxford_flowers | 4,096 | 286 MB | `split_zhou_OxfordFlowers.json` → test |
| oxford_pets | 4,405 | 178 MB | `split_zhou_OxfordPets.json` → test |
| stanford_cars | 9,676 | 461 MB | `split_zhou_StanfordCars.json` → test |
| sun397 | 23,820 | 11 GB | `split_zhou_SUN397.json` → test |
| ucf101 | 5,681 | 149 MB | `split_zhou_UCF101.json` → test |

**Aircraft + OOD variants:**

| Dataset | `new/` images | `new/` size | Filter |
|---|---:|---:|---|
| fgvc_aircraft | 6,666 | 656 MB | `images_variant_test.txt` (val+test) |
| imagenet | 50,000 | — | full set |
| imagenet-adversarial | 7,500 | 213 MB | full set |
| imagenet-rendition | 30,000 | 958 MB | full set |
| imagenet-sketch | 50,889 | 3.6 GB | full set |
| imagenetv2 | 10,000 | 310 MB | full set |
| **Subtotal** | | **~21 GB** | |

QP=7 variants (`new_qp7/`) are intentionally **skipped**.

### 3.2 ret reconstructions  (inside `$RET_DATA_SRC/`)

| Subdir produced | Images | Size | Source |
|---|---:|---:|---|
| `coco2014_cloud/val2014/` | 5,000 | 199 MB | COCO val2014 `karpathy_test` split |
| **Subtotal** | | **~0.2 GB** | |

QP=7 variant (`coco2014_cloud_qp7/`) is intentionally **skipped**.

---

## 4. Recommended upload bundles

A suggested chunking for cloud-drive sharing (one zip per group keeps
download links manageable):

| Bundle | Size | Contents |
|---|---:|---|
| `weights.tar` | ~2.7 GB | CLIP (4 files, shared cls+ret) + enc2dec `model0_qp1.pk` + MaskCLIP pretrain (12 files) |
| `datasets_cls_raw.tar` | ~74 GB | 15 raw cls datasets (§2.1) |
| `datasets_cls_new_qp1.tar` | ~21 GB | 14 `new/` folders (§3.1, 268k reconstructed images) |
| `datasets_ret_raw.tar` | ~22 GB | coco2014 + flickr30k + nocaps (§2.2) |
| `datasets_ret_cloud_qp1.tar` | ~0.2 GB | `coco2014_cloud/val2014/` 5,000 images (§3.2) |
| `datasets_seg.tar` | ~6 GB | ADE20K + VOC2012 (§2.3, ~45k images) |
| **Total** | **~126 GB** | |

If `sun397` (44 GB raw + 11 GB reconstructed) is too large for a single
bundle, split it off into `datasets_cls_sun397.tar`.

---

## 5. .env template for each target

After uploading + downloading to your machine, set these in `.env`:

```bash
CLS_DATA_SRC=/abs/path/to/datasets_cls_raw       # raw + new/ co-located
RET_DATA_SRC=/abs/path/to/datasets_ret_raw       # coco2014 / flickr30k / nocaps + coco2014_cloud
SEG_DATA_SRC=/abs/path/to/datasets_seg           # ade / VOCdevkit
CLIP_CACHE_SRC=/abs/path/to/weights/openai_clip  # shared by cls + ret (auto-symlinked)
ENC2DEC_MODELS_SRC=/abs/path/to/weights/enc2dec  # contains model0_qp1.pk
SEG_PRETRAIN_SRC=/abs/path/to/weights/maskclip_pretrain
# RET_PRETRAINED_SRC is optional — leave unset to reuse CLIP_CACHE_SRC
```

Then run `bash scripts/setup.sh` to wire the symlinks.
