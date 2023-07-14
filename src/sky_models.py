from sky_image_generator import batch_luminance
from abc import ABC, abstractmethod
import numpy as np
import warnings
from coordinates import CoordinateConvertor
from PIL import Image


class SkyModel(ABC):
    """Abstract class for sky models which defines their interface"""
    def __init__(self, W:int, H:int) -> None:
        """Generates a sky model with given resolution

        Args:
            W (int): Width of modelled images
            H (int): Height of modelled images
        """
        self.convertor = CoordinateConvertor(W, H)
        self.default_camera_azimuth = 0
        self.default_sun_azimuth = np.deg2rad(120)
        self.default_sun_zenith = np.deg2rad(80)

    @abstractmethod
    def model_raw(self, theta:np.ndarray, gamma:np.ndarray) -> np.ndarray:
        """Raw run of the model without any coordinate conversion

        Args:
            theta (np.ndarray): 1D array of sky point zenith angles
            gamma (np.ndarray): 1D array of angles between sun direction and sky point direction

        Returns:
            np.ndarray: 1D array of luminance values
        """
        ...

    def model(self, camera_azimuth:float, camera_zenith:float, f:float, sun_azimuth:float, sun_zenith:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Model sky pixels for given camera parameters and sun positions

        Args:
            camera_azimuth (float): in radians in range [0, 2pi], 0=north, pi/2=east, pi=south, 3pi/2=west
            camera_zenith (float): in radians in range [0, pi/2]
            f (float): in pixels, with respect to resolution given in constructor
            sun_azimuth (float): in radians in range [0, 2pi], 0=north, pi/2=east, pi=south, 3pi/2=west
            sun_zenith (float): angle between sun direction and zenith direction (straight up), in range [0, pi/2]
            x (np.ndarray): array of x indices of pixels to be modelled
            y (np.ndarray): array of y indices of pixels to be modelled

        Returns:
            np.ndarray: array of modelled luminance values
        """
        if sun_azimuth is None:
            sun_azimuth = self.default_sun_azimuth
        if sun_zenith is None:
            sun_zenith = self.default_sun_zenith
        if camera_azimuth is None:
            camera_azimuth = self.default_camera_azimuth
            
        theta, gamma = self.convertor.xy_to_theta_gamma(
            x, y, camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith)
        return self.model_raw(theta, gamma)
    
    def generate_image(self, camera_azimuth:float, camera_zenith:float, f:float, sun_azimuth:float, sun_zenith:float) -> np.ndarray:
        """Generates full 2D matrix from the model, should be preprocessed before saving to PNG or JPG file.

        Args:
            camera_azimuth (float): in radians in range [0, 2pi], 0=north, pi/2=east, pi=south, 3pi/2=west
            camera_zenith (float): in radians in range [0, pi/2]
            f (float): in pixels, with respect to resolution given in constructor
            sun_azimuth (float): in radians in range [0, 2pi], 0=north, pi/2=east, pi=south, 3pi/2=west
            sun_zenith (float): angle between sun direction and zenith direction (straight up), in range [0, pi/2]

        Returns:
            np.ndarray: 2D matrix which contains the modelled image
        """
        x, y = np.ones([self.convertor.W, self.convertor.H]).nonzero()
        r = self.model(camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith, x, y)
        if len(r.shape) == 1:
            img = np.zeros([self.convertor.H, self.convertor.W])
            img[y, x] = r
            return img
        else:
            img = np.zeros([self.convertor.H, self.convertor.W, 3])
            img[y, x] = r
            return img
            

class PragueSkyModel(SkyModel):
    """Wrapper for Prague Sky Model implemented in Clang. Needs file SkyModelDataset.dat to work properly. https://drive.google.com/file/d/19K96jEQmmqCeg8yjgZxj2awQj62lI50p/view
    """
    def __init__(self, W:int, H:int, ground_albedo:float=0.00595418177, visibility:float=100.012133) -> None:
        """On top of SkyModel constructor, sets default values for ground albedo and visibility
        Args:
            W (int): image width in pixels
            H (int): image height in pixels
            ground_albedo (float, optional): How much light is reflected from the ground, in range [0,1]. Defaults to 0.00595418177.
            visibility (float, optional): How many kilometers its possible to see. Defaults to 100.012133.
        """
        super().__init__(W, H)
        self.sun_elevation = np.pi/4
        self.albedo = ground_albedo
        self.visibility = visibility

    def model_raw(self, theta:np.ndarray, gamma:np.ndarray) -> np.ndarray:
        """Wrapper for batch_luminance function from sky_image_generator.py which uses Prague Sky Model's default values for ground albedo and visibility

        Args:
            theta (np.ndarray): 1D array of sky point zenith angles
            gamma (np.ndarray): 1D array of angles between sun direction and sky point direction

        Returns:
            np.ndarray: 1D array of luminance values
        """
        return batch_luminance(self.sun_elevation, self.visibility, self.albedo, theta, gamma)

    def model(self, camera_azimuth:float, camera_zenith:float, f:float, sun_azimuth:float, sun_zenith:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Model sky pixels for given camera parameters and sun positions

        Args:
            camera_azimuth (float): in radians in range [0, 2pi], 0=north, pi/2=east, pi=south, 3pi/2=west
            camera_zenith (float): in radians in range [0, pi/2]
            f (float): in pixels, with respect to resolution given in constructor
            sun_azimuth (float): in radians in range [0, 2pi], 0=north, pi/2=east, pi=south, 3pi/2=west
            sun_zenith (float): angle between sun direction and zenith direction (straight up), in range [0, pi/2]
            x (np.ndarray): array of x indices of pixels to be modelled
            y (np.ndarray): array of y indices of pixels to be modelled

        Returns:
            np.ndarray: array of modelled luminance values
        """
        if sun_zenith is None:
            sun_zenith = self.default_sun_zenith
        self.sun_elevation = np.pi/2 - sun_zenith
        value = super().model(camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith, x, y)
        return np.array(value)


class PerezSkyModel(SkyModel):
    """Straightforward implementation of Perez Sky Model
    """
    def __init__(self, W:int, H:int) -> None:
        """Set image resolution

        Args:
            W (int): image width in pixels
            H (int): image height in pixels
        """
        super().__init__(W, H)

    def model_raw(self, theta:np.ndarray, gamma:np.ndarray) -> np.ndarray:
        """Analytically calculates relative luminance with respect to zenith luminance for given sky point angles. 
        Uses default arguments a,b,c,d,e for Perez Sky Model which model clear sky. 

        Args:
            theta (np.ndarray): _description_
            gamma (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        a, b, c, d, e = -1, -0.32, 10, -3, 0.45
        warnings.filterwarnings("ignore", category=RuntimeWarning, message='overflow encountered in exp')

        l_p = (1 + a * np.exp(b / np.cos(theta))) * \
            (1 + c * np.exp(d * gamma) + e * np.cos(gamma)**2)
        return np.nan_to_num(l_p, nan=0, posinf=1000, neginf=-1000)


class PerezAzimuthIndependentSkyModel(SkyModel):
    """Implements Perez Azimuth Independent Sky Model simplified version of Perez Model introduced by J. F. Lalonde https://vision.gel.ulaval.ca/~jflalonde/publications/projects/sky/index.html
    """
    def __init__(self, W:int, H:int) -> None:
        """Set image resolution

        Args:
            W (int): image width in pixels
            H (int): image height in pixels
        """
        super().__init__(W, H)

    def model_raw(self, theta:np.ndarray) -> np.ndarray:
        """Analytical calculation of relative luminance for given sky points' zenith angles

        Args:
            theta (np.ndarray): 1D array of sky point zenith angles

        Returns:
            np.ndarray: 1D array of relative luminance values
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning, message='overflow encountered in exp')
        return np.nan_to_num(1 - np.exp(-0.32/np.cos(theta)), nan=0, posinf=1000, neginf=-1000)
    
    
    def model(self, camera_azimuth:float, camera_zenith:float, f:float, sun_azimuth:float, sun_zenith:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Model sky pixels for given camera zenith. Most parameters are ignored but included for compatibility with other models.

        Args:
            camera_azimuth (float): Ignored
            camera_zenith (float): in radians in range [0, pi/2]
            f (float): in pixels with respect to resolution given in constructor.
            sun_azimuth (float): Ignored
            sun_zenith (float): Ignored
            x (np.ndarray): 1D array of x indices of pixels to be modelled
            y (np.ndarray): 1D array of y indices of pixels to be modelled

        Returns:
            np.ndarray: 1D array of modelled luminance values
        """
        u, v = self.convertor.xy_to_uv(x, y)
        theta = self.convertor.point_zenith(u, v, camera_zenith, f)
        return self.model_raw(theta)
    
    
    def generate_image(self, camera_zenith:float, f:float) -> np.ndarray:
        """Image generation for given camera zenith and focal length, simplified version of original method in SkyModel class

        Args:
            camera_zenith (float): in radians in range [0, pi/2]
            f (float): in pixels with respect to resolution given in constructor.

        Returns:
            np.ndarray: 2D matrix which contains the modelled image
        """
        x, y = np.ones([self.convertor.W, self.convertor.H]).nonzero()
        r = self.model(None, camera_zenith, f, None, None, x, y)
        img = np.zeros([self.convertor.H, self.convertor.W])
        img[y, x] = r
        return img
    

if __name__ == '__main__':
    # Example of sky models usage
    def save_img(model, phi, theta, f, sun_phi, sun_theta, image_name):
        """Demonstration of sky model processing before saving to image
        """
        sky = model.generate_image(phi, theta, f, sun_phi, sun_theta)
        sky[sky < 0] = 0
        sky = sky / np.max(sky)*255
        Image.fromarray(sky.astype(np.uint8)).save(image_name)
        
    sun_phi = 5.275306729916581 
    sun_theta = np.deg2rad(60)
    f = 600
    f_calib = 1377
    theta = np.deg2rad(70)
    phi = sun_phi + 0.4
    perez = PerezSkyModel(720, 540)
    prague = PragueSkyModel(720, 540)
    perez_ind = PerezAzimuthIndependentSkyModel(720, 540)
    
    save_img(prague, phi, theta, f, sun_phi, sun_theta, 'prague.png')
    save_img(perez, phi, theta, f, sun_phi, sun_theta, 'perez.png')
