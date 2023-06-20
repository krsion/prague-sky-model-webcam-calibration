import numpy as np
from PIL import Image


def read_image_greyscale(path):
    return np.array(Image.open(path).convert('L').resize((1600, 1200)))
