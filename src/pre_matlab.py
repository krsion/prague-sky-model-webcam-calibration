import os
from PIL import Image
from scipy.io import savemat
import shutil
import h5py
from sun_position_calculator import SunPositionCalculator




def load_solar(image_path:str)->tuple[float, float]:
    """Calculates sun azimuth and zenith angle from image path. Overriding this method is useful when you want to use different dataset format.

    Args:
        image_path (str): Path to image. It is assumed that there is .mat file with same name in same directory.

    Returns:
        tuple[float, float]: Sun azimuth and zenith angle in radians.
    """
    solar_path = image_path[:-4] + '.mat'
    if not os.path.exists(solar_path):
        return None, None
    mat_file = h5py.File(solar_path, 'r')            
    return mat_file['sunAzimuth'][0][0], mat_file['sunZenith'][0][0]

def pre_matlab():
    suncalc = SunPositionCalculator('../data/webcams.json')
    
    
    PATHIN = '../data/Ismall'
    PATHOUT = '../data-matlab/Ismall'
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


    PATHIN = '../data/Jsmall'
    PATHOUT = '../data-matlab/Jsmall'
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
                    filepathout_mat = os.path.join(PATHOUT, location, str(i) + '.mat')
                    Image.open(filepathin).resize((W, H)).save(filepathout)
                    savemat(filepathout_mat, suncalc.sun_position(filepathin))
                    i += 1
                    
    PATHIN = '../data/masks'
    PATHOUT = '../data-matlab/masks'
    if os.path.exists(PATHOUT):
        shutil.rmtree(PATHOUT)
    os.mkdir(PATHOUT)
    for mask in os.listdir(PATHIN):
        if file.endswith('.jpg'):
            filepathin = f'{PATHIN}/{mask}'
            filepathout = f'{PATHOUT}/{mask}'
            Image.open(filepathin).resize((W, H)).save(filepathout)

pre_matlab()