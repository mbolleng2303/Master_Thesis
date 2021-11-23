import numpy as np


def normalize1(arr):
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
       minval = arr[..., i].min()
    maxval = arr[..., i].max()
    if minval != maxval:
       arr[..., i] -= minval
    arr[..., i] *= (255.0 / (maxval - minval))
    return arr

# z-score normalization
def normalize(arr):
    arr = np.array(arr)
    m = np.mean(arr)
    s = np.std(arr)
    return (arr - m) / s