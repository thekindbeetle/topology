import numpy as np


def fractal_dimension(im, threshold=0.5):
    """
    Фрактальная размерность изображения.
    """
    # Only for 2d image
    assert(len(im.shape) == 2)

    def box_count(Z, k):
        s = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return np.count_nonzero(s)

    # Transform Z into a binary array
    im = (im > threshold)

    # Minimal dimension of image
    p = min(im.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**0)
    sizes = 2**np.arange(n, 0, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(box_count(im, size))

    # Fit 
    coefficients = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coefficients[0]
