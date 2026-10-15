import numpy as np
from digits import split_digits, digit_batch

ink = np.zeros((100, 300))
ink[10:90, 20:40] = 1      # digit 1
ink[10:90, 60:70] = 1      # gap of 20 -> separate digit
ink[10:90, 75:85] = 1      # gap of 5 -> same digit as previous
assert split_digits(ink) == [(20, 40), (60, 85)], split_digits(ink)
assert digit_batch(ink).shape == (2, 28, 28, 1)
assert digit_batch(np.zeros((10, 10))).shape[0] == 0
print("ok")
