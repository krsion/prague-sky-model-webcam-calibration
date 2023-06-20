import os
import shutil

PATHIN = '../data/images'
PATHOUT = '../data/images-matlab'

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
                shutil.copy(filepathin, filepathout)
                i += 1
