"""Run this script to prepare data for the Matlab function chmuCalibration('I', 'J') and chmuCalibration('Ismall', 'Jsmall')"""

import os
from PIL import Image
from scipy.io import savemat
import shutil
import h5py
from sun_position_calculator import SunPositionCalculator

suncalc = SunPositionCalculator('../data/webcams.json')


def copy_images(PATHIN, PATHOUT, W, H, include_solar):
    if os.path.exists(PATHOUT):
        shutil.rmtree(PATHOUT)
    os.makedirs(PATHOUT)
    for location in os.listdir(PATHIN):
        os.makedirs(os.path.join(PATHOUT, location))
        i = 0
        for day in os.listdir(os.path.join(PATHIN, location)):
            for file in os.listdir(os.path.join(PATHIN, location, day)):
                if file.endswith('.jpg'):
                    filepathin = f'{PATHIN}/{location}/{day}/{file}'
                    filepathout = os.path.join(PATHOUT, location, str(i) + '.jpg')
                    Image.open(filepathin).resize((W, H)).save(filepathout)
                    if include_solar:
                        filepathout_mat = os.path.join(PATHOUT, location, str(i) + '.mat')
                        sunpos = suncalc.sun_position(filepathin)
                        sunpos['sunAzimuth'] = - sunpos['sunAzimuth']
                        savemat(filepathout_mat, sunpos)
                    i += 1
    

def pre_matlab():
    W, H = 720, 540
    copy_images('../data/Ismall', '../data-matlab/Ismall', W, H, False)
    copy_images('../data/Jsmall', '../data-matlab/Jsmall', W, H, True)
    copy_images('../data/I', '../data-matlab/I', W, H, False)
    copy_images('../data/J', '../data-matlab/J', W, H, True)
                    
    PATHIN = '../data/masks'
    PATHOUT = '../data-matlab/masks'
    if os.path.exists(PATHOUT):
        shutil.rmtree(PATHOUT)
    os.mkdir(PATHOUT)
    for mask in os.listdir(PATHIN):
        if mask.endswith('.jpg'):
            filepathin = f'{PATHIN}/{mask}'
            filepathout = f'{PATHOUT}/{mask}'
            Image.open(filepathin).resize((W, H)).save(filepathout)

pre_matlab()