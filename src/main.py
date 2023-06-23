import argparse
import numpy as np
from calibration import ArizonaCalibration, CHMUCalibration
from sky_models import PerezSkyModel, PerezAzimuthIndependentSkyModel, PragueSkyModel

parser = argparse.ArgumentParser(description="Lalonde's camera calibration of zenith, azimuth and focal length.")
parser.add_argument('-I', type=str, help='Path to folder containing dataset I used for calibration of focal length and zenith degree.')
parser.add_argument('-J', type=str, help='Path to folder containing dataset J used for calibration of azimuth degree.')
parser.add_argument('-m', '--mask', type=str, help='Path to sky mask image.')
parser.add_argument('-n', '--px-num', type=int, help='Number of pixels in each image to be used for calibration.')
parser.add_argument('-W', type=int, help='Width of images in dataset. Will be resized to this value.')
parser.add_argument('-H', type=int, help='Height of images in dataset. Will be resized to this value.')
parser.add_argument('-f0', type=float, help='Initial guess of focal length.')
parser.add_argument('-df', '--dataset-format', type=str, choices=['date-time', 'matlab-flat'], help='Format of dataset folder.')
parser.add_argument('-mJ', '--model-J', choices=['perez', 'prague'], help='Sky model for dataset J.')

if __name__ == '__main__':
    args = parser.parse_args()
    print('args:', args)
    np.random.seed(0)    
    
    I_model = PerezAzimuthIndependentSkyModel(args.W, args.H)
    J_model = PragueSkyModel(args.W, args.H) if args.model_J == 'prague' else PerezSkyModel(args.W, args.H)
    
    if args.dataset_format == 'date-time':
        CHMUCalibration(args.mask, args.W, args.H, args.px_num).demo(args.I, args.J, args.f0, I_model, J_model)
    else: # matlab-flat
        ArizonaCalibration(args.mask, args.W, args.H, args.px_num).demo(args.I, args.J, args.f0, I_model, J_model)
    