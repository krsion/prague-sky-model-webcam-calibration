import os
import numpy as np
from scipy.optimize import least_squares
from utils import read_image_greyscale
from sun_position_calculator import SunPositionCalculator
from sky_models import PerezAzimuthIndependentSkyModel, PerezSkyModel, PragueSkyModel


class SingleCameraCalibrator:

    def __init__(self, images_path: str, datetime_structure:bool, mask_filename: str, px_num: int, width: int = 720, height: int = 540) -> None:
        """Prepares data for calibration of given camera.

        Args:
            images_path (str): folder with images. Must contain subfolders with images from each day with format images_path/[yyyymmdd]/[HHMM].jpg
            mask_filename (str): mask of the sky, sky is white, rest is black.
            px_num (int): number of random sky pixels in each image to be used for calibration
        """
        self.W = width
        self.H = height
        self._prepare_data(images_path, mask_filename, datetime_structure, px_num)

    def focal_length_and_zenith(self, model: str, verbose=0, f0=500, theta0 : float=None, f_bounds=(10, 5_000), theta_bounds=(0, np.pi*5/8), k_bounds=(-1000, 1000)) -> tuple[float, float]:
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
        if model not in ['psm', 'perez', 'perez-simple']:
            raise ValueError("Model must be either 'psm', 'perez' or 'perez-simple'")
        
        if model == 'psm':
            model = PragueSkyModel(self.W, self.H) 
        elif model == 'perez':
            model=PerezSkyModel(self.W, self.H)
        elif model == 'perez-simple':
            model=PerezAzimuthIndependentSkyModel(self.W, self.H)
        
        if theta0 is None:
            theta0 = self.calc_theta0(f0)

        N_images = len(self.values)

        def objective(x):
            f, theta, ks = x[0], x[1], x[2:]
            residuals = []
            for i in range(N_images):
                modelled = model.model(0, theta, f, np.pi, self.sunZeniths[i], self.xs[i], self.ys[i])
                residual = modelled*ks[i] - self.values[i]
                residuals.append(residual)
            return np.array(residuals).flatten()

        x0 = np.array([f0, theta0, *[1]*N_images])
        print(f0, np.deg2rad(theta0))
        bounds = ((f_bounds[0], theta_bounds[0], *[k_bounds[0]]*N_images),
                  (f_bounds[1], theta_bounds[1], *[k_bounds[1]]*N_images))
        result = least_squares(objective, x0, bounds=bounds, verbose=verbose)

        f, theta = result.x[0], result.x[1]*180/np.pi
        return f, theta

    
    def _prepare_data(self, photos_path: str, mask_filename: str, datetime_structure: bool,  px_num: int = 1000) -> None:
        """From each image in folder photos_path, randomly selects px_num pixels from the sky and stores their values, x and y coordinates. 
           Also stores sun azimuth and zenith for each image. Stores it in attributes self.values, self.xs, self.ys, self.sunAzimuths and self.sunZeniths.
        """
        mask = read_image_greyscale(mask_filename, self.W, self.H)
        self.set_y_max(mask)
        self.values, self.xs, self.ys = [], [], []
        
        if datetime_structure:
            self.sunAzimuths, self.sunZeniths = [], []
            sun_calc = SunPositionCalculator()

            for day in os.listdir(photos_path):
                day_path = os.path.join(photos_path, day)
                for filename in os.listdir(day_path):
                    if not filename.endswith('.jpg'):
                        continue
                    file_path = os.path.join(day_path, filename)
                    img = read_image_greyscale(file_path, self.W, self.H)
                    
                    truth, x, y = self._random_pixels(img, mask, px_num)
                    self.values.append(truth)
                    self.xs.append(x)
                    self.ys.append(y)
                    
                    solar_info = sun_calc.sun_position(file_path)
                    self.sunAzimuths.append(solar_info['sunAzimuth']/180*np.pi)
                    self.sunZeniths.append(solar_info['sunZenith']/180*np.pi)
        else:
            for filename in os.listdir(photos_path):
                if not filename.endswith('.jpg'):
                    continue
                file_path = os.path.join(photos_path, filename)
                img = read_image_greyscale(file_path, self.W, self.H)
                
                truth, x, y = self._random_pixels(img, mask, px_num)
                self.values.append(truth)
                self.xs.append(x)
                self.ys.append(y)
                

    def set_y_max(self, mask):
        y_sky, _ = (mask == 255).nonzero()
        self.y_max = y_sky.max()
        
        
    def calc_theta0(self, f0):
        v_min = 600-self.y_max
        return np.pi/2 + np.arctan2(v_min, f0)
    
    def _random_pixels(self, img: np.ndarray, mask: np.ndarray, px_num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly selects given number of pixels from image that fit the mask.

        Args:
            img (np.ndarray): source image WxH
            mask (np.ndarray): mask image WxH
            px_num (int): number of pixels to return

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: pixel values, x coordinates, y coordinates
        """
        y_sky, x_sky = (mask == 255).nonzero()
        rand_idx = np.random.choice(x_sky.size, px_num)
        y_sky_rand = y_sky[rand_idx]
        x_sky_rand = x_sky[rand_idx]
        truth = img[y_sky_rand, x_sky_rand]/256
        return truth, x_sky_rand, y_sky_rand


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
