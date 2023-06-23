import os
import h5py
import numpy as np
from scipy.optimize import least_squares
from utils import read_image_greyscale, iterable_argument_cache
from sky_models import PerezAzimuthIndependentSkyModel, PerezSkyModel
from sun_position_calculator import SunPositionCalculator



class ArizonaCalibration:
    def __init__(self, mask_path, W, H, px_num):
        mask = read_image_greyscale(mask_path, W, H)
        self.y_sky, self.x_sky = (mask == 255).nonzero()
        self.W = W
        self.H = H
        self.px_num = px_num
                
    def demo(self, I_path, J_path, f0):
        f, theta_c = self.find_f_theta(I_path, f0)
        print(f'f: {f}, theta: {np.rad2deg(theta_c)}', end=', ')
        
        costs, phis = [], []
        for phi0 in np.arange(0, 2*np.pi, np.pi/2):
            phi_c, cost = self.find_phi(J_path, phi0, theta_c, f)
            phis.append(phi_c)
            costs.append(cost)
        
        phi = phis[np.argmin(costs)]
        print('phi:', np.rad2deg(phi))


    def find_f_theta(self, images_path, f0):
        model = PerezAzimuthIndependentSkyModel(self.W, self.H)
        truth, xs, ys, _, _ = self.process_images(self.get_image_paths(images_path))
        theta0 = np.pi/2 + np.arctan2(self.H/2-np.max(self.y_sky), f0)
        x0 = np.array([f0, theta0, *[1]*len(truth)])
        result = least_squares(self.objective_f_theta, x0, method='lm', args=(model, truth, xs, ys))
        f, theta = result.x[0], result.x[1]
        return f, theta


    def find_phi(self, images_path, phi0, theta_c, f):
        model = PerezSkyModel(self.W, self.H)
        truth, xs, ys, sun_phis, sun_thetas = self.process_images(self.get_image_paths(images_path))
        x0 = np.array([phi0, *[1]*len(truth)])
        result = least_squares(self.objective_phi, x0, method='lm', args=(model, truth, xs, ys, theta_c, f, sun_thetas, sun_phis))
        phi = result.x[0]
        return phi, result.cost


    def objective_f_theta(self, x, model, truth, x_idx, y_idx):
        f, theta, ks = x[0], x[1], x[2:]
        residuals = []
        for i in range(len(truth)):
            modelled = model.model(None, theta, f, None, None, x_idx[i], y_idx[i])
            residual = modelled*ks[i] - truth[i]
            residuals.append(residual)
        return np.array(residuals).flatten()

    def objective_phi(self, x, model, truth, x_idx, y_idx, theta_c, f, thetas_s, phis_s):
        phi_c, ks = x[0], x[1:]
        residuals = []
        for i in range(len(truth)):
            modelled = model.model(phi_c, theta_c, f, phis_s[i], thetas_s[i], x_idx[i], y_idx[i])
            residual = modelled*ks[i] - truth[i]
            residuals.append(residual)
        return np.array(residuals).flatten()


    def get_image_paths(self, directory):
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.jpg'):
                yield os.path.join(directory, filename)

    def load_solar(self, image_path):
        solar_path = image_path[:-4] + '.mat'
        if not os.path.exists(solar_path):
            return None, None
        mat_file = h5py.File(solar_path, 'r')            
        return mat_file['sunAzimuth'][0][0], mat_file['sunZenith'][0][0]

    @iterable_argument_cache
    def process_images(self, image_paths):
        truth, xs, ys, sun_phis, sun_thetas = [], [], [], [], []
        
        for image_path in image_paths:
            photo = read_image_greyscale(image_path, self.W, self.H)
            rand_idx = np.random.choice(self.x_sky.size, self.px_num)
            x_sky_rand = self.x_sky[rand_idx]
            y_sky_rand = self.y_sky[rand_idx]
            xs.append(x_sky_rand)
            ys.append(y_sky_rand)
            truth.append(photo[y_sky_rand, x_sky_rand]/256)
            
            phi_s, theta_s = self.load_solar(image_path)           
            sun_phis.append(phi_s)
            sun_thetas.append(theta_s)

        return truth, xs, ys, sun_phis, sun_thetas
            


class CHMUCalibration(ArizonaCalibration):
    def __init__(self, mask_path, W, H, px_num):
        self.solar_calc = SunPositionCalculator()
        super().__init__(mask_path, W, H, px_num)
        
    def load_solar(self, image_path):
        x = self.solar_calc.sun_position(image_path)
        return np.deg2rad(x['sunAzimuth']), np.deg2rad(x['sunZenith'])
    
    def get_image_paths(self, directory):
        for day in os.listdir(directory):
            for filename in os.listdir(os.path.join(directory, day)):
                if filename.endswith('.jpg'):
                    yield os.path.join(directory, day, filename)


if __name__ == '__main__':
    np.random.seed(0)
    PX_NUM, F0, WIDTH, HEIGHT = 1000, 500, 720, 540
    #ArizonaCalibration('../webcamCalibration/skyMask/mask.jpg', WIDTH, HEIGHT, PX_NUM).demo('../webcamCalibration/images/gradient', '../webcamCalibration/images/clearDay', F0)
    CHMUCalibration('../data/sky-masks/ceske_budejovice.jpg', WIDTH, HEIGHT, PX_NUM).demo('../data/I/ceske_budejovice', '../data/J/ceske_budejovice', F0)