import os
import shutil
from PIL import Image
import numpy as np
from calibration import CHMUCalibration
from scipy.io import savemat

PATHIN = '../data/I'
PATHOUT = '../data-matlab/I'

W, H = 720, 540

if os.path.exists(PATHOUT):
    shutil.rmtree(PATHOUT)

os.mkdir(PATHOUT)



for location in os.listdir(PATHIN):
    os.mkdir(os.path.join(PATHOUT, location))
    i = 0
    for day in os.listdir(os.path.join(PATHIN, location)):
        for file in os.listdir(os.path.join(PATHIN, location, day)):
            if file.endswith('.jpg'):
                filepathin = f'{PATHIN}/{location}/{day}/{file}'
                filepathout = os.path.join(PATHOUT, location, str(i) + '.jpg')
                Image.open(filepathin).resize((W, H)).save(filepathout)
                i += 1


PATHIN = '../data/J'
PATHOUT = '../data-matlab/J'

if os.path.exists(PATHOUT):
    shutil.rmtree(PATHOUT)

os.mkdir(PATHOUT)

calib = CHMUCalibration('../data/sky-masks/ceske_budejovice.jpg', W, H, 1000)
for location in os.listdir(PATHIN):
    os.mkdir(os.path.join(PATHOUT, location))
    i = 0
    for day in os.listdir(os.path.join(PATHIN, location)):
        for file in os.listdir(os.path.join(PATHIN, location, day)):
            if file.endswith('.jpg'):
                filepathin = f'{PATHIN}/{location}/{day}/{file}'
                filepathout = os.path.join(PATHOUT, location, str(i) + '.jpg')
                filepathout_mat = os.path.join(PATHOUT, location, str(i) + '.mat')

                Image.open(filepathin).resize((W, H)).save(filepathout)
                phi, theta = calib.load_solar(filepathin)
                savemat(filepathout_mat, {'sunAzimuth': phi, 'sunZenith': theta})

                i += 1
                
PATHIN = '../data/sky-masks'
PATHOUT = '../data-matlab/sky-masks'

if os.path.exists(PATHOUT):
    shutil.rmtree(PATHOUT)

os.mkdir(PATHOUT)

for mask in os.listdir(PATHIN):
    if file.endswith('.jpg'):
        filepathin = f'{PATHIN}/{mask}'
        filepathout = f'{PATHOUT}/{mask}'
        Image.open(filepathin).resize((W, H)).save(filepathout)
                
