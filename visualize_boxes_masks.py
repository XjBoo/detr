"""
Minimal inference + visualization script:
- loads a (panoptic) DETR model that predicts boxes + masks
- runs forward on one image (or all images in a directory)
- overlays instance-like masks and draws boxes

Examples:
  # Single image
  python visualize_boxes_masks.py \
    --image_path /path/to/image.jpg \
    --output_path /tmp/detr_vis.png \
    --device cuda

  # Directory batch
  python visualize_boxes_masks.py \
    --image_path testPic \
    --output_path picTesOutput \
    --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TVF

import hubconf
from models.detr import PostProcess


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resize_like_detr(img: Image.Image, min_size: int = 800, max_size: int = 1333) -> Image.Image:
    # Same policy as datasets/transforms.py::resize() with scalar min_size.
    w, h = img.size
    min_original_size = float(min(w, h))
    max_original_size = float(max(w, h))

    size = min_size
    if (max_original_size / min_original_size) * size > max_size:
        size = int(round(max_size * min_original_size / max_original_size))

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return img.resize((ow, oh), resample=Image.BILINEAR)


def preprocess_image(img: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Returns:
      image_tensor: FloatTensor [3,H,W], normalized
      orig_size_hw: (H_orig, W_orig) in pixels (after resize, before padding)
    """
    img = img.convert("RGB")
    img = _resize_like_detr(img, min_size=800, max_size=1333)
    w, h = img.size
    image_tensor = TVF.to_tensor(img)  # [0,1], CHW
    image_tensor = TVF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return image_tensor, (h, w)


def _random_color_from_id(idx: int) -> Tuple[int, int, int]:
    # Deterministic "random" color per idx.
    rng = np.random.default_rng(seed=idx + 12345)
    r, g, b = (rng.integers(0, 255, size=3)).tolist()
    return int(r), int(g), int(b)


def overlay_mask_rgba(base_rgba: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
    """
    base_rgba: HxWx4 uint8, modified in-place
    mask: HxW bool
    """
    if not mask.any():
        return
    r, g, b = color
    overlay = np.zeros_like(base_rgba, dtype=np.float32)
    overlay[..., 0] = r
    overlay[..., 1] = g
    overlay[..., 2] = b
    overlay[..., 3] = 255

    base = base_rgba.astype(np.float32)
    m = mask[..., None].astype(np.float32)
    base_rgba[:] = (base * (1.0 - m * alpha) + overlay * (m * alpha)).astype(np.uint8)


def draw_boxes(
    img: Image.Image,
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    score_thresh: float,
) -> None:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(boxes_xyxy.shape[0]):
        s = float(scores[i].item())
        if s < score_thresh:
            continue
        x0, y0, x1, y1 = boxes_xyxy[i].tolist()
        color = _random_color_from_id(int(i))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        text = f"id={int(labels[i].item())} score={s:.2f}"
        if font is not None:
            draw.text((x0 + 2, y0 + 2), text, fill=color, font=font)
        else:
            draw.text((x0 + 2, y0 + 2), text, fill=color)


def iter_image_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"--image_path not found: {path}")
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files: List[Path] = []
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No images found in directory: {path}")
    return files


def resolve_output_file(output_path: Path, img_file: Path) -> Path:
    # If output_path looks like a file path, keep it (single-image mode).
    if output_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp") and not output_path.is_dir():
        return output_path
    # Directory mode: write <stem>.png under output_path.
    return output_path / f"{img_file.stem}.png"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("DETR box+mask visualization (single image or directory)")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model",
        type=str,
        default="detr_resnet50_panoptic",
        choices=(
            "detr_resnet50_panoptic",
            "detr_resnet50_dc5_panoptic",
            "detr_resnet101_panoptic",
        ),
        help="Panoptic models predict pred_masks; good for mask overlay.",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Optional: load a local checkpoint (.pth)")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=50, help="Max number of queries to visualize")
    parser.add_argument("--mask_alpha", type=float, default=0.45)
    args = parser.parse_args()

    device = torch.device(args.device)
    input_path = Path(args.image_path)
    output_path = Path(args.output_path)
    image_files = iter_image_files(input_path)

    # Build model (pretrained weights by default).
    model_fn = getattr(hubconf, args.model)
    model = model_fn(pretrained=True)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    post_bbox = PostProcess()

    # Ensure output directory exists.
    if output_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    for img_file in image_files:
        img_pil = Image.open(img_file)
        img_tensor, (h, w) = preprocess_image(img_pil)

        samples = img_tensor.unsqueeze(0).to(device)  # [1,3,H,W]
        outputs = model(samples)  # dict with pred_logits, pred_boxes, pred_masks

        results = post_bbox(outputs, target_sizes=torch.tensor([[h, w]], device=device))
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        boxes = results[0]["boxes"]  # [num_queries,4] in pixels

        topk = min(args.topk, scores.numel())
        topk_idx = torch.topk(scores, k=topk, sorted=True).indices
        scores = scores[topk_idx]
        labels = labels[topk_idx]
        boxes = boxes[topk_idx]

        if "pred_masks" not in outputs:
            raise RuntimeError(
                "Model output has no 'pred_masks'. Use a panoptic/segmentation model (e.g. detr_resnet50_panoptic)."
            )
        pred_masks = outputs["pred_masks"][0, topk_idx]  # [topk,Hm,Wm]
        pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        pred_masks = pred_masks.sigmoid()

        base_vis = img_pil.convert("RGB")
        base_vis = _resize_like_detr(base_vis, min_size=800, max_size=1333)
        base_rgba = np.array(base_vis.convert("RGBA"), dtype=np.uint8)

        for i in range(topk):
            if float(scores[i].item()) < args.score_thresh:
                continue
            m = pred_masks[i] > args.mask_thresh
            color = _random_color_from_id(int(i))
            overlay_mask_rgba(base_rgba, m.detach().cpu().numpy(), color=color, alpha=float(args.mask_alpha))

        vis_img = Image.fromarray(base_rgba, mode="RGBA").convert("RGB")
        draw_boxes(vis_img, boxes.detach().cpu(), labels.detach().cpu(), scores.detach().cpu(), args.score_thresh)

        out_file = resolve_output_file(output_path, img_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        vis_img.save(out_file)
        print(f"[ok] {img_file.name} -> {out_file}")


if __name__ == "__main__":
    main()

