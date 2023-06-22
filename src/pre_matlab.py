import os
import shutil
from PIL import Image

PATHIN = '../data/images'
PATHOUT = '../data/images-matlab'

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
