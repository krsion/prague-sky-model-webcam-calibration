import argparse
import json
import numpy as np
from calibration import ArizonaCalibration, CHMUCalibration
from sky_models import PerezSkyModel, PerezAzimuthIndependentSkyModel, PragueSkyModel

parser = argparse.ArgumentParser(description="Lalonde's camera calibration of zenith, azimuth and focal length.")
parser.add_argument('-I', type=str, help='Path to folder containing dataset I used for calibration of focal length and zenith angle.')
parser.add_argument('-J', type=str, help='Path to folder containing dataset J used for calibration of azimuth angle.')
parser.add_argument('-m', '--mask', type=str, help='Path to sky mask image.')
parser.add_argument('-n', '--px-num', type=int, help='Number of pixels in each image to be used for calibration.')
parser.add_argument('-W', type=int, help='Width of images in dataset. Will be resized to this value.')
parser.add_argument('-H', type=int, help='Height of images in dataset. Will be resized to this value.')
parser.add_argument('-f0', type=float, help='Initial guess of focal length.')
parser.add_argument('-df', '--dataset-format', type=str, choices=['date-time', 'matlab-flat'], help='Format of dataset folder.')
parser.add_argument('-mI', '--model-I', choices=['perez', 'perez-azimuth-independent', 'prague'], help='Sky model for dataset I.')
parser.add_argument('-mJ', '--model-J', choices=['perez', 'prague'], help='Sky model for dataset J.')
parser.add_argument('-min', type=int, help='Pixel values lower than this will be ignored for calibration.')
parser.add_argument('-max', type=int, help='Pixel values higher than this will be ignored for calibration.')
parser.add_argument('-webcams', type=str, help='Path to JSON file containing webcam positions. Only used when --dataset-format=date-time')

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(0)
    
    I_model = PragueSkyModel(args.W, args.H)
    J_model = PragueSkyModel(args.W, args.H) if args.model_J == 'prague' else PerezSkyModel(args.W, args.H)
    
    if args.dataset_format == 'date-time':
        results = CHMUCalibration(args.mask, args.W, args.H, args.px_num, args.min, args.max, args.webcams).demo(args.I, args.J, args.f0, I_model, J_model)
        results['args'] = vars(args)
        print(json.dumps(results))
    else: # matlab-flat
        results = ArizonaCalibration(args.mask, args.W, args.H, args.px_num, args.min, args.max).demo(args.I, args.J, args.f0, I_model, J_model)
        results['args'] = vars(args)
        print(json.dumps(results))