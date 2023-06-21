import numpy as np
from PIL import Image


def read_image_greyscale(path, width, height):
    return np.array(Image.open(path).convert('L').resize((width, height)))
