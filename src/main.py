import argparse
import json
import os
from calibrator import SingleCameraCalibrator


parser = argparse.ArgumentParser(
    description='Lalonde camera calibration of zenith degree and focal length using Prague Sky Model or Perez Sky Model')
parser.add_argument('-p', '--photos', default='../data/images', type=str,
                    help='Folder containing clear sky photos with structure like \
                    PHOTOS/[location]/[yyyymmdd]/[HHMM].jpg (default: %(default)s)')
parser.add_argument('-d', '--datetime-structure', action='store_true', default=False, type=bool, help='if PHOTOS/[location]/[yyyymmdd]/[HHMM].jpg. Otherways PHOTOS/*.jpg')
parser.add_argument('-m', '--masks', default='../data/sky-masks', type=str,
                    help='Folder containing sky masks named [location].jpg (default: %(default)s)')
parser.add_argument('-s', '--sky-model', default='psm', type=str, choices=['psm', 'perez, perez-simple'],
                    help='Sky model to be used. (default: %(default)s)')
parser.add_argument('-o', '--output', type=str,
                    help='File path to output results in json format. (default: stdout)')
parser.add_argument('-n', '--px-num', default=1000, type=int,
                    help='Number of pixels in each image to be used for calibration. (default: %(default)s)')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    results = {}
    for location in os.listdir(args.photos):
        print(location)
        location_path = os.path.join(args.photos, location)
        mask_path = os.path.join(args.masks, location + '.jpg')
        calib = SingleCameraCalibrator(location_path, mask_path, args.px_num)
        f, theta = calib.focal_length_and_zenith(args.sky_model)
        results[location] = {'focalLength': f, 'zenithDegree': theta}

    if args.output:
        directory = os.path.dirname(args.output)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(args.output, 'w') as file:
            json.dump(results, file)
    else:
        print(results)
