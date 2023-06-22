import argparse
import json
import os
import numpy as np
from calibrator import SingleCameraCalibrator
from sky_models import PragueSkyModel, PerezSkyModel, PerezAzimuthIndependentSkyModel


parser = argparse.ArgumentParser(
    description='Lalonde camera calibration of zenith degree and focal length using Prague Sky Model or Perez Sky Model')
parser.add_argument('-p', '--photos', default='../data/images', type=str,
                    help='Folder containing clear sky photos with structure like \
                    PHOTOS/[location]/[yyyymmdd]/[HHMM].jpg (default: %(default)s)')
parser.add_argument('-m', '--masks', default='../data/sky-masks', type=str,
                    help='Folder containing sky masks named [location].jpg (default: %(default)s)')
parser.add_argument('-s', '--sky-model', default='psm', type=str, choices=['psm', 'perez', 'perez-simple'],
                    help='Sky model to be used. (default: %(default)s)')
parser.add_argument('-o', '--output', type=str,
                    help='File path to output results in json format. (default: stdout)')
parser.add_argument('-n', '--px-num', default=1000, type=int,
                    help='Number of pixels in each image to be used for calibration. (default: %(default)s)')
parser.add_argument('--solar', action='store_true', help='Use solar zenith angle for optimization', default=False)

if __name__ == '__main__':
    W, H = 720, 540
    args = parser.parse_args()
    print('ARGS', args)
    
    model = None
    if args.sky_model == 'psm':
        model = PragueSkyModel(W, H) 
    elif args.sky_model == 'perez':
        model = PerezSkyModel(W, H)
    else:
        model = PerezAzimuthIndependentSkyModel(W, H)
    
    
    results = {}
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