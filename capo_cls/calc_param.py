import os
import torch
import clip

cache_dir = os.environ.get(
    'CLIP_CACHE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clip_cache')
)
cache_dir = os.path.expanduser(cache_dir)
name_map = {
    "RN50.pt": "RN50",
    "RN101.pt": "RN101",
    "ViT-B-16.pt": "ViT-B/16",
    "ViT-L-14.pt": "ViT-L/14"
}

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# 输出 CSV 表头
print("Model,Image encoder params,Text encoder params,Total params")

for fname, model_name in name_map.items():
    model_path = os.path.join(cache_dir, fname)
    if not os.path.exists(model_path):
        continue

    loaded = clip.load(model_name, device="cpu", download_root=cache_dir)
    model = loaded[0] if isinstance(loaded, tuple) else loaded

    img_params = count_params(model.visual)
    txt_params = count_params(model.transformer)
    total_params = img_params + txt_params

    print(f"{model_name},{img_params},{txt_params},{total_params}")
