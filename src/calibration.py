import os
import h5py
import numpy as np
from scipy.optimize import least_squares
from utils import read_image_greyscale, iterable_argument_cache
from sun_position_calculator import SunPositionCalculator
from sky_models import SkyModel

class ArizonaCalibration:
    """
    Optimization of focal length, zenith and azimuth angle on a flat dataset.
    """
    def __init__(self, mask_path:str, W:int, H:int, px_num:int):
        """Initialize ArizonaCalibration.

        Args:
            mask_path (str): Sky mask image path. Sky pixels are white, other pixels are black.
            W (int): Width of images in dataset. Will be resized to this value.
            H (int): Height of images in dataset. Will be resized to this value.
            px_num (int): Number of pixels in each image to be used for calibration.
        """
        mask = read_image_greyscale(mask_path, W, H)
        self.y_sky, self.x_sky = (mask == 255).nonzero()
        self.W = W
        self.H = H
        self.px_num = px_num
                
    def demo(self, I_path:str, J_path:str, f0:int, I_model:SkyModel, J_model:SkyModel) -> dict[str, float, float]:
        """Run calibration on dataset I and J. Calibration of azimuth angle is done for each of the four cardinal directions.
        Args:
            I_path (str): Path to folder containing dataset I used for calibration of focal length and zenith degree.
            J_path (str): Path to folder containing dataset J used for calibration of azimuth degree.
            f0 (int): Initial guess of focal length.
            I_model (SkyModel): Instance of SkyModel used for calibration over dataset I.
            J_model (SkyModel): Instance of SkyModel used for calibration over dataset J.

        Returns:
            dict[str, float, float]: Calibration results, angles converted to degrees.
        """
        f, theta_c = self.find_f_theta(I_path, f0, I_model)
        costs, phis = [], []
        for phi0 in np.arange(0, 2*np.pi, np.pi/2):
            phi_c, cost = self.find_phi(J_path, phi0, theta_c, f, J_model)
            phis.append(phi_c)
            costs.append(cost)
        
        phi = phis[np.argmin(costs)]
        return {'f':f, 'theta':np.rad2deg(theta_c) % 360, 'phi':np.rad2deg(phi) % 360}


    def find_f_theta(self, images_path:str, f0:int, model:SkyModel) -> tuple[float, float]:
        """Optimization of focal length and zenith angle.

        Args:
            images_path (str): Path to folder containing dataset I used for calibration.
            f0 (int): Initial guess of focal length.
            model (SkyModel): Instance of SkyModel used for calibration.

        Returns:
            tuple[float, float]: focal length in pixels and zenith angle in radians.
        """
        truth, xs, ys, _, _ = self.process_images(self.get_image_paths(images_path))
        theta0 = np.pi/2 + np.arctan2(self.H/2-np.max(self.y_sky), f0)
        x0 = np.array([f0, theta0, *[1]*len(truth)])
        result = least_squares(self.objective_f_theta, x0, method='lm', args=(model, truth, xs, ys))
        f, theta = result.x[0], result.x[1]
        return f, theta


    def find_phi(self, images_path:str, phi0:int, theta_c:float, f:int, model:SkyModel) -> tuple[float, float]:
        """Optimization of azimuth angle

        Args:
            images_path (str): Path to folder containing dataset J used for calibration.
            phi0 (int): Initial guess of azimuth angle.
            theta_c (float): Zenith angle in radians.
            f (int): Focal length in pixels.
            model (SkyModel): Instance of SkyModel used azimuth angle estimation.

        Returns:
            tuple[float, float]: Azimuth angle in radians and value of loss function.
        """
        truth, xs, ys, sun_phis, sun_thetas = self.process_images(self.get_image_paths(images_path))
        x0 = np.array([phi0, *[1]*len(truth)])
        result = least_squares(self.objective_phi, x0, method='lm', args=(model, truth, xs, ys, theta_c, f, sun_thetas, sun_phis))
        phi = result.x[0]
        return phi, result.cost


    def objective_f_theta(self, x:np.ndarray, model:SkyModel, truth:list[np.ndarray], x_idx:list[int], y_idx:list[int]) -> np.ndarray:
        """Calculates residuals from which is optimized mean square error for non-linear least squares.

        Args:
            x (np.ndarray): wrapped f, theta and each image's scale factors
            model (SkyModel): generator of sky model images
            truth (list[np.ndarray]): values of selected sky pixels
            x_idx (list[int]): x coordinates of selected sky pixels
            y_idx (list[int]): y coordinates of selected sky pixels

        Returns:
            np.ndarray: calculated residuals
        """
        f, theta, ks = x[0], x[1], x[2:]
        residuals = []
        for i in range(len(truth)):
            modelled = model.model(None, theta, f, None, None, x_idx[i], y_idx[i])
            residual = modelled*ks[i] - truth[i]
            residuals.append(residual)
        return np.array(residuals).flatten()

    def objective_phi(self, x:np.ndarray, model:SkyModel, truth:list[np.ndarray], x_idx:list[np.ndarray], y_idx:list[np.ndarray], theta_c: float, f:float, thetas_s:list[float], phis_s:list[float]) -> np.ndarray:
        """Calculates residuals from which is optimized mean square error for non-linear least squares.

        Args:
            x (np.ndarray): wrapped azimuth angle and each image's scale factors
            model (SkyModel): generator of sky model images
            truth (list[np.ndarray]): values of selected sky pixels
            x_idx (list[np.ndarray]): x coordinates of selected sky pixels
            y_idx (list[np.ndarray]): y coordinates of selected sky pixels
            theta_c (float): zenith angle in radians of camera used for sky model generation
            f (float): focal length in pixels of camera used for sky model generation
            thetas_s (list[float]): sun zenith angle in radians for each image
            phis_s (list[float]): sun azimuth angle in radians for each image

        Returns:
            np.ndarray: calculated residuals
        """
        phi_c, ks = x[0], x[1:]
        residuals = []
        for i in range(len(truth)):
            modelled = model.model(phi_c, theta_c, f, phis_s[i], thetas_s[i], x_idx[i], y_idx[i])
            residual = modelled*ks[i] - truth[i]
            residuals.append(residual)
        return np.array(residuals).flatten()


    def get_image_paths(self, directory:str):
        """Iterator over image paths in directory. Overriding this method is useful when you want to use different dataset format.

        Args:
            directory (str): root directory of dataset

        Yields:
            str: paths to images to be used for calibration
        """
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.jpg'):
                yield os.path.join(directory, filename)

    def load_solar(self, image_path:str)->tuple[float, float]:
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

    @iterable_argument_cache
    def process_images(self, image_paths:list[str])->tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float], list[float]]:
        """Prepares data for optimization

        Args:
            image_paths (list[str]): List of paths to images used for calibration

        Returns:
            Tuple of lists. Each list item is for one image:
            truth:      list[np.ndarray] ... pixel values
            xs:         list[np.ndarray] ... pixel x coordinates,
            ys:         list[np.ndarray] ... pixel y coordinates,
            sun_phis:   list[float]      ... sun azimuths in radians
            sun_thetas: list[float]      ... sun zeniths in radians
        """
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
    """Child class of ArizonaCalibration. It is used for CHMU dataset format. Sun positions are calculated from image path. e.g. I/brno/20220531/1345.jpg
    """
    def __init__(self, mask_path:str, W:int, H:int, px_num:int):
        """Creates instance of CHMUCalibration with preset mask, image size and number of pixels used for calibration.

        Args:
            mask_path (str): path to mask image
            W (int): Width of images
            H (int): Height of images
            px_num (int): Number of pixels used for calibration
        """
        self.solar_calc = SunPositionCalculator()
        super().__init__(mask_path, W, H, px_num)
        
    def load_solar(self, image_path):
        """Calculates sun azimuth and zenith angle from image path. Overriding this method is useful when you want to use different dataset format.
        Calculates sun position from datetime of capture and location of camera.
        
        Args:
            image_path (str): Path to image. It is assumed that it's name is in format location/yyyymmdd/HHMM.jpg
        Returns:
            _type_: _description_
        """
        x = self.solar_calc.sun_position(image_path)
        return np.deg2rad(x['sunAzimuth']), np.deg2rad(x['sunZenith'])
    
    def get_image_paths(self, directory):
        """Overriden method from ArizonaCalibration. Iterates over image paths in directory tree.

        Args:
            directory (str): root directory of dataset

        Yields:
            str: paths to images to be used for calibration
        """
        for day in os.listdir(directory):
            for filename in os.listdir(os.path.join(directory, day)):
                if filename.endswith('.jpg'):
                    yield os.path.join(directory, day, filename)


if __name__ == '__main__':
    np.random.seed(0)
    PX_NUM, F0, WIDTH, HEIGHT = 1000, 500, 720, 540
    #ArizonaCalibration('../webcamCalibration/skyMask/mask.jpg', WIDTH, HEIGHT, PX_NUM).demo('../webcamCalibration/images/gradient', '../webcamCalibration/images/clearDay', F0)
    print(CHMUCalibration('../data/sky-masks/ceske_budejovice.jpg', WIDTH, HEIGHT, PX_NUM).demo('../data/I/ceske_budejovice', '../data/J/ceske_budejovice', F0))