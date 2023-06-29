import numpy as np
from PIL import Image
import functools

def iterable_argument_cache(func):
    """Cache decorator for functions with argument being a list (mostly of strings).
    """
    memo = {}

    @functools.wraps(func)
    def wrapper(*args):
        key = tuple(args)  # Convert list of strings to a tuple
        if key not in memo:
            memo[key] = func(*args)
        return memo[key]

    return wrapper

def read_image_greyscale(path: str, width:int, height:int) -> np.ndarray:
    """Unified way of reading greyscale images. 

    Args:
        path (str): path to image
        width (int): width to resize image to
        height (int): height to resize image to

    Returns:
        np.ndarray: HxW matrix of greyscale values
    """
    return np.array(Image.open(path).convert('L').resize((width, height)))
