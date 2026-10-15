"""Draw a number (any digits, side by side) and see it read back."""
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras
from digits import digit_batch

W, H, BRUSH = 640, 220, 16
model = keras.models.load_model("models/digits.keras")

root = tk.Tk()
root.title("Number recognizer")
canvas = tk.Canvas(root, width=W, height=H, bg="black", cursor="cross")
canvas.pack(padx=10, pady=10)
result = tk.Label(root, text="Draw a number", font=("Segoe UI", 28))
result.pack()
detail = tk.Label(root, text="", font=("Segoe UI", 11))
detail.pack()

img = Image.new("L", (W, H))
pen = ImageDraw.Draw(img)
last = [None]


def draw(e):
    if last[0]:
        canvas.create_line(*last[0], e.x, e.y, fill="white", width=BRUSH, capstyle="round")
        pen.line([*last[0], e.x, e.y], fill=255, width=BRUSH)
    last[0] = (e.x, e.y)


def predict(_=None):
    last[0] = None
    batch = digit_batch(np.asarray(img, dtype="float32") / 255.0)
    if not len(batch):
        return
    p = model.predict(batch, verbose=0)
    result.config(text="".join(str(d) for d in p.argmax(1)))
    detail.config(text="   ".join(f"{c:.0%}" for c in p.max(1)))


def clear():
    canvas.delete("all")
    pen.rectangle([0, 0, W, H], fill=0)
    result.config(text="Draw a number")
    detail.config(text="")


canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", predict)
tk.Button(root, text="Clear", command=clear).pack(pady=8)
root.mainloop()
