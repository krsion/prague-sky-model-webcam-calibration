import numpy as np
from PIL import Image
import functools

def iterable_argument_cache(func):
    memo = {}

    @functools.wraps(func)
    def wrapper(*args):
        key = tuple(args)  # Convert list of strings to a tuple
        if key not in memo:
            memo[key] = func(*args)
        return memo[key]

    return wrapper

def read_image_greyscale(path, width, height):
    return np.array(Image.open(path).convert('L').resize((width, height)))
