import numpy as np
import cv2 as cv
def contrast_stretching(img):
    """Apply contrast stretching to the input image."""
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    in_min = np.percentile(img, 5)
    in_max = np.percentile(img, 95)
    out_min = 0.0
    out_max = 255.0
    out = (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
    out[out < out_min] = out_min
    out[out > out_max] = out_max
    return out.astype(np.uint8)