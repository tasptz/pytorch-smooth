import numpy as np

def sobel(k:int):
    '''
    Generate a horizontal sobel kernel with shape `(k, k)`
    '''
    if k % 2 == 0:
        y, x = np.mgrid[:k, :k] + 0.5 - k / 2
    else:
        y, x = np.mgrid[:k, :k] - k // 2
    norm = np.square(x) + np.square(y)
    norm[norm == 0.] = 1.
    f = x / norm
    # construct img with gradient gx = 1
    i = np.tile(np.arange(k)[None], (k, 1))
    # response
    r = (f * i).sum()
    # normalize to response = 1
    f = f / r
    return f
