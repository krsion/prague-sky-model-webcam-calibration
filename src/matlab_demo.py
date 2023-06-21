import os
import numpy as np
from utils import read_image_greyscale
from sky_models import PerezAzimuthIndependentSkyModel, PerezSkyModel, PragueSkyModel
from scipy.optimize import least_squares



def matlab_demo(photos_path, mask_path, px_num, f0, model, W, H):
    mask = read_image_greyscale(mask_path, W, H)
    y_sky, x_sky = (mask == 255).nonzero()

    
    truth, xs, ys = [], [], []
    for photo_name in os.listdir(photos_path):
        photo = read_image_greyscale(os.path.join(photos_path, photo_name), W, H)

        rand_idx = np.random.choice(x_sky.size, px_num)
        x_sky_rand = x_sky[rand_idx]
        y_sky_rand = y_sky[rand_idx]

        xs.append(x_sky_rand)
        ys.append(y_sky_rand)
        truth.append(photo[y_sky_rand, x_sky_rand]/256)
    
    N = len(truth)

    def objective(x):
        f, theta, ks = x[0], x[1], x[2:]
        residuals = []
        for i in range(N):
            modelled = model.model(None, theta, f, None, None, xs[i], ys[i])
            residual = modelled*ks[i] - truth[i]
            residuals.append(residual)
        return np.array(residuals).flatten()
    
    theta0 = np.pi/2 + np.arctan2(H/2-np.max(y_sky), f0)
    print('f0:', f0, 'theta0:', np.rad2deg(theta0))

    
    x0 = np.array([f0, theta0, *[1]*N])
    result = least_squares(objective, x0, method='lm')
    print('f:', result.x[0], 'theta:', np.rad2deg(result.x[1]))
    
W, H = 720, 540
print('perez azimuth independent')
matlab_demo('../webcamCalibration/images/gradient', '../webcamCalibration/skyMask/mask.jpg', 1000, 500, PerezAzimuthIndependentSkyModel(W, H), W, H)

print('perez')
matlab_demo('../webcamCalibration/images/gradient', '../webcamCalibration/skyMask/mask.jpg', 1000, 500, PerezSkyModel(W, H), W, H)

print('prague')
matlab_demo('../webcamCalibration/images/gradient', '../webcamCalibration/skyMask/mask.jpg', 1000, 500, PragueSkyModel(W, H), W, H)

