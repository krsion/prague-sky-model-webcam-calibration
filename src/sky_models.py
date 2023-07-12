from sky_image_generator import batch_luminance
from abc import ABC, abstractmethod
import numpy as np
import warnings
from coordinates import CoordinateConvertor
from PIL import Image


class SkyModel(ABC):
    def __init__(self, W:int, H:int) -> None:
        self.convertor = CoordinateConvertor(W, H)
        self.default_camera_azimuth = 0
        self.default_sun_azimuth = np.deg2rad(120)
        self.default_sun_zenith = np.deg2rad(80)

    @abstractmethod
    def model_raw(self, theta:np.ndarray, gamma:np.ndarray) -> np.ndarray:
        ...

    def model(self, camera_azimuth:float, camera_zenith:float, f:float, sun_azimuth:float, sun_zenith:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
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
    def __init__(self, W:int, H:int, ground_albedo:float=0.00595418177, visibility:float=100.012133) -> None:
        super().__init__(W, H)
        self.sun_elevation = np.pi/4
        self.albedo = ground_albedo
        self.visibility = visibility

    def model_raw(self, theta:np.ndarray, gamma:np.ndarray) -> np.ndarray:
        return batch_luminance(self.sun_elevation, self.visibility, self.albedo, theta, gamma)

    def model(self, camera_azimuth:float, camera_zenith:float, f:float, sun_azimuth:float, sun_zenith:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        if sun_zenith is None:
            sun_zenith = self.default_sun_zenith
        self.sun_elevation = np.pi/2 - sun_zenith
        value = super().model(camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith, x, y)
        return np.array(value)


class PerezSkyModel(SkyModel):
    def __init__(self, W:int, H:int) -> None:
        super().__init__(W, H)

    def model_raw(self, theta:np.ndarray, gamma:np.ndarray) -> np.ndarray:
        a, b, c, d, e = -1, -0.32, 10, -3, 0.45
        warnings.filterwarnings("ignore", category=RuntimeWarning, message='overflow encountered in exp')

        l_p = (1 + a * np.exp(b / np.cos(theta))) * \
            (1 + c * np.exp(d * gamma) + e * np.cos(gamma)**2)
        return np.nan_to_num(l_p, nan=0, posinf=1000, neginf=-1000)


class PerezAzimuthIndependentSkyModel(SkyModel):
    def __init__(self, W, H) -> None:
        super().__init__(W, H)

    def model_raw(self, theta:np.ndarray) -> np.ndarray:
        warnings.filterwarnings("ignore", category=RuntimeWarning, message='overflow encountered in exp')
        return np.nan_to_num(1 - np.exp(-0.32/np.cos(theta)), nan=0, posinf=1000, neginf=-1000)
    
    
    def model(self, camera_azimuth:float, camera_zenith:float, f:float, sun_azimuth:float, sun_zenith:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        u, v = self.convertor.xy_to_uv(x, y)
        theta = self.convertor.point_zenith(u, v, camera_zenith, f)
        return self.model_raw(theta)
    
    
    def generate_image(self, camera_zenith:float, f:float):
        x, y = np.ones([self.convertor.W, self.convertor.H]).nonzero()
        r = self.model(None, camera_zenith, f, None, None, x, y)
        img = np.zeros([self.convertor.H, self.convertor.W])
        img[y, x] = r
        return img
    

if __name__ == '__main__':
    def save_img(model, phi, theta, f, sun_phi, sun_theta, image_name):
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
