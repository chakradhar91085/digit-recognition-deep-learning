# Multi-Digit Number Recognizer

Draw a number (two digits or more) on a canvas and a CNN reads it back, with a confidence per digit.

## How it works

1. `train.py` trains a small CNN on MNIST (with rotation/shift/zoom augmentation) and saves `models/digits.keras`.
2. `app.py` captures your drawing, finds digits by scanning for vertical gaps between strokes (`digits.py`), reshapes each to MNIST's 28x28 layout, and classifies them one by one.

## Run

Use Python 3.10-3.12 (TensorFlow does not support 3.14 yet).

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
mkdir models
python train.py
python app.py
python test_digits.py   # quick check of the segmentation logic
```

Leave a clear gap between digits; touching digits are read as one.

## Credits

Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/). Built with TensorFlow/Keras, Pillow and Tkinter. MIT licensed.
