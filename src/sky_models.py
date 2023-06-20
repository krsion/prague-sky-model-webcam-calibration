from sky_image_generator import batch_luminance

import numpy as np
from coordinates import CoordinateConvertor


class SkyModel:
    def __init__(self, W, H) -> None:
        self.convertor = CoordinateConvertor(W, H)

    def model_raw(self, theta, gamma):
        ...

    def model(self, camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith, x, y):
        theta, gamma = self.convertor.xy_to_theta_gamma(
            x, y, camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith)
        return self.model_raw(theta, gamma)
    
    def generate_image(self, camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith):
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
    def __init__(self, W=1600, H=1200, ground_albedo=0.00595418177, visibility=100.012133) -> None:
        super().__init__(W, H)
        self.sun_elevation = np.pi/4
        self.albedo = ground_albedo
        self.visibility = visibility

    def model_raw(self, theta, gamma):
        return batch_luminance(self.sun_elevation, self.visibility, self.albedo, theta, gamma)

    def model(self, camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith, x, y):
        self.sun_elevation = np.pi/2 - sun_zenith
        value = super().model(camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith, x, y)
        return np.array(value)


class PerezSkyModel(SkyModel):
    def __init__(self, W, H) -> None:
        super().__init__(W, H)

    def model_raw(self, theta, gamma):
        a, b, c, d, e = -1, -0.32, 10, -3, 0.45
        l_p = (1 + a * np.exp(b / np.cos(theta))) * \
            (1 + c * np.exp(d * gamma) + e * np.cos(gamma)**2)
        return np.nan_to_num(l_p)


class PerezAzimuthIndependentSkyModel(SkyModel):
    def __init__(self, W, H) -> None:
        self.convertor = CoordinateConvertor(W, H)

    def model_raw(self, theta, gamma):
        return np.nan_to_num(1 - np.exp(-0.32/np.cos(theta)))
