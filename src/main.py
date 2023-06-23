import argparse
import json
import os
import numpy as np
from calibration import ArizonaCalibration, CHMUCalibration

parser = argparse.ArgumentParser(description="Lalonde's camera calibration of zenith, azimuth and focal length.")
parser.add_argument('-I', type=str, help='Path to folder containing dataset I used for calibration of focal length and zenith degree.')
parser.add_argument('-J', type=str, help='Path to folder containing dataset J used for calibration of azimuth degree.')
parser.add_argument('-m', '--mask', type=str, help='Path to sky mask image.')
parser.add_argument('-n', '--px-num', type=int, help='Number of pixels in each image to be used for calibration.')
parser.add_argument('-W', type=int, help='Width of images in dataset. Will be resized to this value.')
parser.add_argument('-H', type=int, help='Height of images in dataset. Will be resized to this value.')
parser.add_argument('-f0', type=float, help='Initial guess of focal length.')
parser.add_argument('-df', '--dataset-format', type=str, choices=['date-time', 'matlab-flat'], help='Format of dataset folder.')

if __name__ == '__main__':
    args = parser.parse_args()
    print('args:', args)
    np.random.seed(0)    
    if args.dataset_format == 'date-time':
        CHMUCalibration(args.mask, args.W, args.H, args.px_num).demo(args.I, args.J, args.f0)
    else:
        ArizonaCalibration(args.mask, args.W, args.H, args.px_num).demo(args.I, args.J, args.f0)
    
    
    
    
    
    '''results = {}
    for location in os.listdir(args.photos):
        location_path = os.path.join(args.photos, location)
        mask_path = os.path.join(args.masks, location + '.jpg')
        calib = SingleCameraCalibrator(location_path, mask_path, args.px_num, 500, W, H, args.solar)
        f, theta = calib.focal_length_and_zenith(model, 0)
        fov = np.rad2deg(model.convertor.f_to_fov(f))
        print(location, 'fov:', fov, 'deg, theta:', theta, 'deg')
        results[location] = {'focalLength': f, 'zenithDegree': theta}

    if args.output:
        directory = os.path.dirname(args.output)
        if len(directory) > 0 and not os.path.exists(directory):
            os.makedirs(directory)
        with open(args.output, 'w') as file:
            json.dump(results, file)
    else:
        print(results)
    '''