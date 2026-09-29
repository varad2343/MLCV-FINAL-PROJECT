"""Pretrained ResNet-18 poster embeddings used by offline and web recommenders."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


MODEL_ID = "resnet18-imagenet1k-v1"
EMBEDDING_DIM = 512


def poster_fingerprint(posters: Iterable[tuple[int, Path]]) -> str:
    """Return a stable cache key for the poster files and embedding model."""
    digest = hashlib.sha256(MODEL_ID.encode("utf-8"))
    for movie_id, path in sorted(posters, key=lambda item: item[0]):
        try:
            stat = path.stat()
            digest.update(f"{movie_id}:{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
        except OSError:
            digest.update(f"{movie_id}:{path.name}:missing\n".encode("utf-8"))
    return digest.hexdigest()


def load_embedding_archive(path: Path) -> tuple[dict[int, np.ndarray], str | None, str | None]:
    """Load an embedding cache without importing PyTorch."""
    path = Path(path)
    if not path.is_file():
        return {}, None, None
    try:
        with np.load(path, allow_pickle=False) as data:
            ids = data["movie_ids"].astype(np.int64)
            vectors = data["embeddings"].astype(np.float32)
            model_id = str(data["model_id"].item()) if "model_id" in data else None
            fingerprint = str(data["fingerprint"].item()) if "fingerprint" in data else None
        if vectors.ndim != 2 or vectors.shape != (len(ids), EMBEDDING_DIM):
            return {}, None, None
        return {int(movie_id): vector for movie_id, vector in zip(ids, vectors)}, model_id, fingerprint
    except (OSError, ValueError, KeyError):
        return {}, None, None


def extract_poster_visual_features(image_path: Path) -> dict[str, float]:
    """Extract interpretable poster properties for a user's visual taste profile."""
    from PIL import Image, ImageFilter

    with Image.open(image_path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        small = rgb.resize((128, 192))
        pixels = np.asarray(small, dtype=np.float32)
        gray = np.asarray(small.convert("L"), dtype=np.float32)
        hsv = np.asarray(small.convert("HSV"), dtype=np.float32)

        red_green = np.abs(pixels[:, :, 0] - pixels[:, :, 1])
        yellow_blue = np.abs(0.5 * (pixels[:, :, 0] + pixels[:, :, 1]) - pixels[:, :, 2])
        colorfulness = np.sqrt(np.var(red_green) + np.var(yellow_blue)) + 0.3 * np.sqrt(
            np.mean(red_green) ** 2 + np.mean(yellow_blue) ** 2
        )
        edge_pixels = np.asarray(small.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)

    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "colorfulness": float(colorfulness),
        "edge_complexity": float(edge_pixels.std()),
        "saturation": float(hsv[:, :, 1].mean() / 255.0),
        "aspect_ratio": float(width / height) if height else 0.0,
    }


def extract_resnet18_embeddings(
    posters: list[tuple[int, Path]],
    batch_size: int = 24,
    device: str | None = None,
) -> tuple[list[int], np.ndarray, list[tuple[int, str]]]:
    """Extract normalized 512-D vectors from local poster files.

    Torch and TorchVision are optional runtime dependencies.  The model is loaded
    only when this function is called, and the trained classification head is
    replaced with identity to return its 512-dimensional pooled representation.
    """
    try:
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:  # TorchVision can fail at import if its wheel is mismatched.
        raise RuntimeError(
            "Deep poster embeddings need compatible torch and torchvision packages. "
            "Install the optional requirements in requirements-vision.txt."
        ) from exc

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(run_device)
    preprocess = weights.transforms()

    movie_ids: list[int] = []
    embeddings: list[np.ndarray] = []
    failures: list[tuple[int, str]] = []

    with torch.inference_mode():
        for offset in range(0, len(posters), batch_size):
            tensors = []
            batch_ids = []
            for movie_id, image_path in posters[offset : offset + batch_size]:
                try:
                    with Image.open(image_path) as opened:
                        tensors.append(preprocess(opened.convert("RGB")))
                    batch_ids.append(int(movie_id))
                except Exception as exc:
                    failures.append((int(movie_id), str(exc)))
            if not tensors:
                continue

            batch = torch.stack(tensors).to(run_device)
            vectors = functional.normalize(model(batch).float(), p=2, dim=1)
            embeddings.extend(vectors.cpu().numpy().astype(np.float32))
            movie_ids.extend(batch_ids)

    matrix = np.vstack(embeddings) if embeddings else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return movie_ids, matrix, failures

