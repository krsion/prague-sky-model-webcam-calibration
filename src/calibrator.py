import os
import numpy as np
from scipy.optimize import least_squares
from utils import read_image_greyscale
from sun_position_calculator import SunPositionCalculator
from sky_models import SkyModel


class SingleCameraCalibrator:

    def __init__(self, images_path: str, mask_filename: str, px_num: int, f0: float, width: int, height: int, solar: bool) -> None:
        self.W = width
        self.H = height
        
        mask = read_image_greyscale(mask_filename, self.W, self.H)
        y_sky, x_sky = (mask == 255).nonzero()
        
        self.f0 = f0
        self.theta0 =  np.pi/2 + np.arctan2(height/2-np.max(y_sky), f0)
        
        self.values, self.xs, self.ys = [], [], []
        self.sunZeniths = []
        sun_calc = SunPositionCalculator()

        for day in os.listdir(images_path):
            day_path = os.path.join(images_path, day)
            for filename in os.listdir(day_path):
                if not filename.endswith('.jpg'):
                    continue
                file_path = os.path.join(day_path, filename)
                img = read_image_greyscale(file_path, self.W, self.H)
                
                rand_idx = np.random.choice(x_sky.size, px_num)
                y_sky_rand = y_sky[rand_idx]
                x_sky_rand = x_sky[rand_idx]
                self.xs.append(x_sky_rand)
                self.ys.append(y_sky_rand)
                self.values.append(img[y_sky_rand, x_sky_rand]/256)
                
                solar_info = sun_calc.sun_position(file_path)
                if solar:
                    self.sunZeniths.append(solar_info['sunZenith']/180*np.pi)
                else:
                    self.sunZeniths.append(None)

    def focal_length_and_zenith(self, model: SkyModel, verbose) -> tuple[float, float]:
        """Uses non-linear optimization to find focal length and zenith angle of the camera

        Args:
            model (str): either 'psm' or 'perez'
            verbose (int, optional): How detailed should be logs from the optimizer. Can be 0, 1, 2 or 3. Defaults to 0.
            f0 (int, optional): Initial focal length. Defaults to 2000.
            theta0 (float, optional): Initial zenith angle. Defaults to 3/8*np.pi.
            f_bounds (tuple, optional): Lower and upper bounds of focal length. Defaults to (10, 20000).
            theta_bounds (tuple, optional): Lower and upper bounds of zenith angle. Defaults to (0, np.pi/2).
            k_bounds (tuple, optional): Lower and upper bounds for scale factor of modelled images. Defaults to (-100, 100).

        Raises:
            ValueError: If model is not 'psm' or 'perez'

        Returns:
            tuple[float, float]: focal length and zenith angle of the camera
        """

        N_images = len(self.values)

        def objective(x):
            f, theta, ks = x[0], x[1], x[2:]
            residuals = []
            for i in range(N_images):
                modelled = model.model(None, theta, f, None, self.sunZeniths[i], self.xs[i], self.ys[i])
                residual = modelled*ks[i] - self.values[i]
                residuals.append(residual)
            return np.array(residuals).flatten()

        x0 = np.array([self.f0, self.theta0, *[1]*N_images])
        #print('fov0', np.rad2deg(model.convertor.f_to_fov(self.f0)), 'deg, theta0', np.rad2deg(self.theta0), 'deg')
        result = least_squares(objective, x0, method='lm', verbose=verbose)

        f, theta = result.x[0], result.x[1]*180/np.pi
        return f, theta


def azimuth(location, f, camZenith, mode='psm', px_num=1000, degrees=True, verbose=1):
    default_azimuths = {'churanov': 0.7853981633974483, 'ceske_budejovice': 4.71238898038469, 'belotin': 1.5707963267948966, 'cheb': 0.7853981633974483, 'brno': 0.7853981633974483,
                        'broumov': 0.7853981633974483, 'dukovany': 2.356194490192345, 'dylen': 5.105088062083414, 'doksany': 3.9269908169872414, 'frydlant': 3.9269908169872414}
    truth, xs, ys, sunAzimuths, sunZeniths = prepare_data(
        'clearskies2', location, luminance_only=True, solar=True)
    # f, camZenith = load_f_zenith(location)
    N = len(truth)

    model = None
    if mode == 'perez':
        model = PerezSkyModel(1600, 1200)
    elif mode == 'psm':
        model = PragueSkyModel(1600, 1200, 0.00595418177, 100.012133)
    else:
        return

    def objective(x):
        azimuth, ks = x[0], x[1:]
        residuals = []
        for i in range(N):
            modelled = model.model(
                azimuth, camZenith, f, sunAzimuths[i], sunZeniths[i], xs[i], ys[i])
            residuals.append(modelled*ks[i] - truth[i])
        return np.array(residuals).flatten()

    x0 = np.array([default_azimuths[location], *[1]*N])
    bounds = (0, *[-1000]*N), (2*np.pi, *[1000]*N)

    result = least_squares(objective, x0, bounds=bounds, verbose=verbose)

    azimuth = result.x[0]
    if degrees:
        azimuth *= 180 / np.pi
    return {'azimuth': azimuth}


def azimuth_on_dataset(name, prev_results, **kwargs):
    results = prev_results
    for location in os.listdir(f'../data/{name}'):
        f = prev_results[location]['focalLength']
        theta_c = prev_results[location]['zenithAngle'] * np.pi/180
        print(location)
        results[location]['azimuthAngle'] = azimuth(location, f, theta_c, **kwargs)['azimuth']
    return results
