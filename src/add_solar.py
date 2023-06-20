import os
from sun_position_calculator import SunPositionCalculator
import scipy.io

PATHIN = '../data/I-train'

sun_calc = SunPositionCalculator()
for location in os.listdir(PATHIN):
    for day in os.listdir(os.path.join(PATHIN, location)):
        for file in os.listdir(os.path.join(PATHIN, location, day)):
            if file.endswith('.jpg'):
                filepathin = f'{PATHIN}/{location}/{day}/{file}'
                solar_path = f'{PATHIN}/{location}/{day}/{file[:-4]}.mat'
                solar_data = sun_calc.sun_position(filepathin)
                scipy.io.savemat(solar_path, solar_data)
