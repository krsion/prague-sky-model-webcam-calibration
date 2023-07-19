import numpy as np
from numpy import sin, cos, pi
import json
from coordinates import CoordinateConvertor
from scipy.optimize import least_squares
from sun_position_calculator import SunPositionCalculator



class PerspectiveCalibrator:
    """Given 4 points in 3D space and their projections on the image plane, calculates camera's parameters.
    """
    def __init__(self, image_width, image_height) -> None:
        self.W = image_width
        self.H = image_height
        pass

    def _P(self, x:list[float], y:list[float], z:list[float], u:list[float], v:list[float]) -> np.ndarray:
        """From 4 points in 3D space and their projections on the image plane calculates the projection matrix P.
        Args:
            x (list[float]): x coordinates of the points in 3D space
            y (list[float]): y coordinates of the points in 3D space
            z (list[float]): z coordinates of the points in 3D space
            u (list[float]): x coordinates of the points on the image plane, with origin in the center of the image.
            v (list[float]): y coordinates of the points on the image plane, with origin in the center of the image.

        Returns:
            np.ndarray: Projection matrix P with shape 8x8
        """
        assert len(x) == len(y) == len(z) == len(u) == len(v) == 4
        rows = []
        for i in range(len(x)):
            rows.append([x[i], y[i], 0, 0, 0, -u[i]*x[i], -u[i]*y[i], -u[i]*z[i]])
            rows.append([0, 0, x[i], y[i], z[i], -v[i]*x[i], -v[i]*y[i], -v[i]*z[i]])
        return np.array(rows)


    def _m(self, P:np.ndarray) -> np.ndarray:
        """Using Singular Value Decomposition of the projection matrix P calculates the vector m from which it is possible to calculate cameras intrinsic and extrinsic parameters.

        Args:
            P (np.ndarray): Projection 8x8 matrix from space to image plane

        Returns:
            np.ndarray: vector mfrom which it is possible to calculate cameras intrinsic and extrinsic parameters
        """
        _, s, Vt = np.linalg.svd(P, full_matrices=True)
        min_singular_index = np.argmin(s)
        x_min = Vt[min_singular_index, :]
        return x_min


    def _raw_calib(self, m:np.ndarray) -> tuple[float, float, float]:
        """From vector m calculates cameras intrinsic and extrinsic parameters.

        Args:
            m (np.ndarray): solves least squares of projection matrix P from space to image plane

        Returns:
            tuple[float, float, float]: camera zenith angle, azimuth angle and focal length
        """
        c = np.sqrt(m[-3]**2 + m[-2]**2 + m[-1]**2)
        m = m / c
        theta_c = np.arctan(np.sqrt(m[-3]**2 + m[-2]**2) / m[-1])
        f = np.sqrt(m[0]**2 + m[1]**2)
        phi_c = np.arctan2(m[0], (-m[1]))
        phi_c %= 2*pi
        return theta_c, phi_c, f


    def _finetuned_calib(self, sun_thetas: list[float], sun_phis: list[float], xs:list[float], ys:list[float]) -> tuple[float, float, float]:
        """First calculates camera parameters using linear least squares and than finetunes them using non-linear least squares.
        Args:
            sun_thetas (list[float]): list of 4 sun zenith angles
            sun_phis (list[float]): list of 4 sun azimuth angles
            xs (list[float]): list of 4 x coordinates of the suns projections on the image plane
            ys (list[float]): list of 4 y coordinates of the suns projections on the image plane

        Returns:
            tuple[float, float, float]: camera zenith angle theta_c, azimuth angle phi_c and focal length f_c 
        """
        convertor = CoordinateConvertor(self.W, self.H)
        x, y, z = convertor.spherical_to_cartesian(sun_thetas, sun_phis)
        u, v = convertor.xy_to_uv(xs, ys)

        def objective_function(params):
            theta_c, phi_c, f = params
            m1 = np.array([f*sin(phi_c), -f*cos(phi_c), 0])
            m2 = np.array([-f*cos(phi_c)*cos(theta_c), -f*sin(phi_c)*cos(theta_c), f*sin(theta_c)])
            m3 = np.array([cos(phi_c)*sin(theta_c), sin(phi_c)*sin(theta_c), cos(theta_c)])
            residuals = []
            for i in range(len(sun_thetas)):
                s = np.array([x[i], y[i], z[i]])
                residuals.append(m1@s - u[i]*m3@s)
                residuals.append(m2@s - v[i]*m3@s)
            return residuals

        x0 = self._raw_calib(self._m(self._P(x, y, z, u, v)))
        x = least_squares(objective_function, x0).x
        return x0, x


    def calibrate(self, sun_thetas:list[float], sun_phis:list[float], xs:list[float], ys:list[float]) -> tuple[float, float, float]:
        """From suns zenith and azimuth angles and their projections on the image plane calculates camera zenith angle, azimuth angle and focal length.

        Args:
            sun_thetas (list[float]): 4 sun zenith angles in radians
            sun_phis (list[float]): 4 sun azimuth angles in radians
            xs (list[float]): 4 sun x coordinates on the image plane
            ys (list[float]): 4 sun y coordinates on the image plane

        Returns:
            tuple[float, float, float]: camera zenith angle theta_c and azimuth angle phi_c in radians and focal length f_c in pixels
        """

        # TODO: remove default and hardcoded values

        thetas_calib, phis_calib, fs_calib = [], [], []
        R = 100
        max_tries = 10_000_000
        num_tries = 0
        counter = 0
        while counter < 2 and num_tries < max_tries:
            num_tries += 1
            xs_moved = xs + np.random.randint(-R, R + 1, 4)
            ys_moved = ys + np.random.randint(-R, R + 1, 4)
            x0, (theta_calib, phi_calib, f_calib) = self._finetuned_calib(sun_thetas, sun_phis, xs_moved, ys_moved)
            if num_tries % 100 == 0:
                ...
                #print('num_tries', num_tries, x0, theta_calib, phi_calib, f_calib)
            if 0 < theta_calib < pi/2 and 100 < f_calib < 8000: 
                #print(num_tries, xs_moved, ys_moved, 'calib theta, phi, f:',
                #    np.rad2deg(theta_calib), np.rad2deg(phi_calib), f_calib)
                counter += 1
                thetas_calib.append(theta_calib)
                phis_calib.append(phi_calib)
                fs_calib.append(f_calib)
                if counter == 2:
                    return np.mean(thetas_calib), np.mean(phis_calib), np.mean(fs_calib)
        if num_tries == max_tries:
            print('max tries reached')
            return thetas_calib, phis_calib, fs_calib
        thetas_calib = np.array(thetas_calib)
        phis_calib = np.array(phis_calib)
        fs_calib = np.array(fs_calib)
        print('counter', counter)
        return np.mean(thetas_calib), np.mean(phis_calib), np.mean(fs_calib)



if __name__ == '__main__':
    np.random.seed(0)
    sun_calc = SunPositionCalculator('../data/webcams.json')
    
    data_json = json.load(open('../data/p4p.json'))
    data_W, data_H, data =data_json['W'], data_json['H'], data_json['locations']
    perspective_calib = PerspectiveCalibrator(data_W, data_H)

    for location in data:
        if location in [ 'belotin']:
            continue
        solar_data = [sun_calc.sun_position(f) for f in data[location]['filenames']]
        data[location]['sun_thetas'] = [x['sunZenith'] for x in solar_data]
        data[location]['sun_phis'] = [x['sunAzimuth'] for x in solar_data]
        theta, phi, f = perspective_calib.calibrate(data[location]['sun_thetas'], data[location]['sun_phis'],
                                    np.array(data[location]['xs']), np.array(data[location]['ys']))
        print(f'{{"name": "{location}", "f": {f}, "theta": {np.rad2deg(theta)}, "phi": {np.rad2deg(phi)}}}')
