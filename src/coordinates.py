import numpy as np
from numpy import sin, cos, pi


class CoordinateConvertor:
    """Used coordinate systems are:
    - (x, y) - 2D image coordinates, origin in the left upper corner
    - (u, v) - 2D image coordinates, origin in the center of the image
    - (theta, phi) - spherical coordinates - angular coordinates of the point in the sky taken from the camera
        - theta - zenith angle
        - phi - azimuth angle
        - gamma - angle between the sun and the point in the sky
    - (x, y, z) - 3D cartesian coordinates, origin in the center of the image
    """

    def __init__(self, W, H) -> None:
        self.W = W
        self.H = H

    def xy_to_uv(self, x: int, y: int) -> tuple[int, int]:
        return x-self.W//2, self.H//2 - y

    def uv_to_xy(self, u: int, v: int) -> tuple[int, int]:
        return u-self.W//2, self.H//2 - v

    def point_zenith(self, u: int, v: int, camera_zenith: float, f: float):
        return np.arccos((v * np.sin(camera_zenith) + f * np.cos(camera_zenith)) /
                         (np.sqrt(f * f + u * u + v * v)))

    def point_azimuth(self, u, v, phi_c, theta_c, f):
        y = f * np.sin(phi_c) * np.sin(theta_c) - u * np.cos(phi_c) - v * np.sin(phi_c) * np.cos(theta_c)
        x = f * np.cos(phi_c) * np.sin(theta_c) + u * np.sin(phi_c) - v * np.cos(phi_c) * np.cos(theta_c)
        return np.arctan2(y, x)

    def point_gamma(self, point_zenith, point_azimuth, sun_azimuth, sun_zenith):
        return np.arccos(np.cos(point_zenith) * np.cos(sun_zenith) +
                         np.sin(point_zenith) * np.sin(sun_zenith) * np.cos(point_azimuth - sun_azimuth))

    def uv_to_theta_phi(self, u, v, camera_azimuth, camera_zenith, f):
        theta = self.point_zenith(u, v, camera_zenith, f)
        phi = self.point_azimuth(u, v, camera_azimuth, camera_zenith, f)
        return theta, phi

    def xy_to_theta_gamma(self, x, y, camera_azimuth, camera_zenith, f, sun_azimuth, sun_zenith):
        u, v = self.xy_to_uv(x, y)
        theta, phi = self.uv_to_theta_phi(u, v, camera_azimuth, camera_zenith, f)
        gamma = self.point_gamma(theta, phi, sun_azimuth, sun_zenith)
        return theta, gamma

    def spherical_to_cartesian(self, theta, phi):
        return np.array([sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])

    def theta_phi_to_uv(self, theta_c, phi_c, f, theta_p, phi_p):
        s = self.spherical_to_cartesian(theta_p, phi_p)
        K = np.array([[0, -f, 0],
                      [0, 0, f],
                      [1, 0, 0]])
        tht = pi/2-theta_c
        Ry = np.array([[cos(tht), 0, sin(tht)],
                       [0, 1, 0],
                       [-sin(tht), 0, cos(tht)]])
        Rz = np.array([[cos(phi_c), sin(phi_c), 0],
                       [-sin(phi_c), cos(phi_c), 0],
                       [0, 0, 1]])
        result = K@Ry@Rz@s
        return np.round(result[:2]/result[-1]).astype(int)
    
    def f_to_fov(self, f):
        return 2*np.arctan2(self.W,2*f)
