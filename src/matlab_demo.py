import os
import numpy as np
from utils import read_image_greyscale
from sky_models import PerezAzimuthIndependentSkyModel, PerezSkyModel, PragueSkyModel
from scipy.optimize import least_squares
import h5py


def matlab_demo(I_path, J_path, mask_path, px_num, f0, W, H):
    mask = read_image_greyscale(mask_path, W, H)
    y_sky, x_sky = (mask == 255).nonzero()

    model_I = PerezAzimuthIndependentSkyModel(W, H)
    model_J = PerezSkyModel(W, H)
    
    truth_I, xs_I, ys_I = [], [], []
    
    for photo_name in os.listdir(I_path):
        photo = read_image_greyscale(os.path.join(I_path, photo_name), W, H)

        rand_idx = np.random.choice(x_sky.size, px_num)
        x_sky_rand = x_sky[rand_idx]
        y_sky_rand = y_sky[rand_idx]

        xs_I.append(x_sky_rand)
        ys_I.append(y_sky_rand)
        truth_I.append(photo[y_sky_rand, x_sky_rand]/256)
        
    truth_J, xs_J, ys_J = [], [], []
    phis_s, thetas_s  = [], []
    
    for filename in sorted(os.listdir(J_path)):
        if filename.endswith('.jpg'):
            photo = read_image_greyscale(os.path.join(J_path, filename), W, H)
            
            rand_idx = np.random.choice(x_sky.size, px_num)
            x_sky_rand = x_sky[rand_idx]
            y_sky_rand = y_sky[rand_idx]
            
            xs_J.append(x_sky_rand)
            ys_J.append(y_sky_rand)
            truth_J.append(photo[y_sky_rand, x_sky_rand]/256)
            
        if filename.endswith('.mat'):
            mat_file = h5py.File(os.path.join(J_path, filename), 'r')            
            phis_s.append(mat_file['sunAzimuth'][0][0])
            thetas_s.append(mat_file['sunZenith'][0][0])
            
                
    N_I = len(truth_I)
    N_J = len(truth_J)

    def objective_I(x):
        f, theta, ks = x[0], x[1], x[2:]
        residuals = []
        for i in range(N_I):
            modelled = model_I.model(None, theta, f, None, None, xs_I[i], ys_I[i])
            residual = modelled*ks[i] - truth_I[i]
            residuals.append(residual)
        return np.array(residuals).flatten()
    
    theta0 = np.pi/2 + np.arctan2(H/2-np.max(y_sky), f0)
    print('f0:', f0, 'theta0:', np.rad2deg(theta0))

    
    x0_I = np.array([f0, theta0, *[1]*N_I])
    result_I = least_squares(objective_I, x0_I, method='lm')
    
    f = result_I.x[0]
    theta_c = result_I.x[1]
    
    print('f:', f, 'theta:', np.rad2deg(theta_c))
    
    def objective_J(x):
        phi_c, ks = x[0], x[1:]
        residuals = []
        for i in range(N_J):
            modelled = model_J.model(phi_c, theta_c, f, phis_s[i], thetas_s[i], xs_J[i], ys_J[i])
            residual = modelled*ks[i] - truth_J[i]
            residuals.append(residual)
        return np.array(residuals).flatten()
    
    costs, phis = [], []
    
    bounds_J = ([-2*np.pi, *[0]*N_J], [np.pi*2, *[np.inf]*N_J])
    
    for phi0 in np.arange(0, 2*np.pi, np.pi/2):
        x0_J = np.array([phi0, *[1]*N_J])
        result_J = least_squares(objective_J, x0_J, bounds=bounds_J)
        phi_c = result_J.x[0]
        phis.append(phi_c)
        costs.append(result_J.cost)
        # print(np.rad2deg(phi_c), result_J.cost)
    
    phi = phis[np.argmin(costs)]
    print('phi:', np.rad2deg(phi))
    
        

W, H = 720, 540

np.random.seed(0)

matlab_demo('../webcamCalibration/images/gradient', '../webcamCalibration/images/clearDay', '../webcamCalibration/skyMask/mask.jpg', 1000, 500, W, H)