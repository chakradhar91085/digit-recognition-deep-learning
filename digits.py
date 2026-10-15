"""Pure-numpy helpers: split a drawing into digits and shape each for the classifier."""
import numpy as np
from PIL import Image


def split_digits(ink, min_gap=12, thresh=0.1):
    """ink: 2D float array, 1 = stroke. Returns [(x0, x1), ...] left to right."""
    cols = ink.max(axis=0) > thresh
    runs, start = [], None
    for x, on in enumerate(np.append(cols, False)):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append([start, x])
            start = None
    merged = []
    for r in runs:  # glue strokes of one digit (e.g. a gappy 4 or 5)
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def to_mnist(ink):
    """Crop to the stroke, fit in 20x20, centre on a 28x28 canvas (MNIST layout)."""
    ys, xs = np.nonzero(ink > 0.1)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20 / max(h, w)
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    out = Image.new("L", (28, 28))
    out.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    return np.asarray(out, dtype="float32")[..., None] / 255.0


def digit_batch(ink):
    """Whole drawing -> (N, 28, 28, 1) array, one row per detected digit."""
    segs = split_digits(ink)
    return np.stack([to_mnist(ink[:, a:b]) for a, b in segs]) if segs else np.empty((0, 28, 28, 1))
