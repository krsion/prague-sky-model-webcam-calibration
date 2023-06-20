import unittest
import numpy as np
from sky_models import PragueSkyModel, PerezSkyModel
from coordinates import CoordinateConvertor
from sun_geometry import calibrate


class TestSkyModels(unittest.TestCase):
    def test_perez_sky_model(self):
        x, y = np.ones([160, 160]).nonzero()

        m = PerezSkyModel(160, 160)
        v0 = m.model(0, np.pi/4, 80, 0, np.pi/4, x, y)
        v360 = m.model(0, np.pi/4, 80, np.pi*2, np.pi/4, x, y)

        self.assertTrue(np.allclose(v0, v360))

    def test_prague_sky_model(self):
        x, y = np.ones([160, 160]).nonzero()

        m = PragueSkyModel(160, 160, 0.1, 100)
        v0 = m.model(0, np.pi/4, 80, 0, np.pi/4, x, y)
        v360 = m.model(0, np.pi/4, 80, np.pi*2, np.pi/4, x, y)

        self.assertTrue(np.allclose(v0, v360))


class TestCoordinateConvertor(unittest.TestCase):
    def setUp(self):
        self.W, self.H = 1600, 1200
        self.convertor = CoordinateConvertor(self.W, self.H)
        self.pixel_positions = [(800, 600), (0, 0), (211, 150), (820, 150), (211, 1131), (820, 1031)]
        self.cam_azimuth_zenith = np.deg2rad([(90, 90), (45, 45), (10, 90), (115, 0), (280, 60), (190, 30)])
        self.f = [2000, 1000, 800, 1200, 1600, 2000]
        self.tolerance = 1e-6

    def test_xy_to_phi_theta_gamma_360_equals_0_degress(self):
        c = CoordinateConvertor(160, 160)
        x, y = np.ones([c.W, c.H]).nonzero()
        u, v = c.xy_to_uv(x, y)
        theta0, gamma0 = c.xy_to_theta_gamma(x, y, 0, np.pi/4, 80, 0, np.pi/4)
        phi0 = np.rad2deg(c.point_azimuth(u, v, 0, np.pi/4, 80))
        theta360, gamma360 = c.xy_to_theta_gamma(x, y, 0, np.pi/4, 80, np.pi*2, np.pi/4)
        phi360 = np.rad2deg(c.point_azimuth(u, v, np.pi*2, np.pi/4, 80))

        self.assertTrue(np.allclose(phi0, phi360))
        self.assertTrue(np.array_equal(theta0, theta360))
        self.assertTrue(np.allclose(gamma0, gamma360))

    def test_uv_to_thetaphi_to_uv(self):
        for cam_pos, f in zip(self.cam_azimuth_zenith, self.f):
            phi_c, theta_c = cam_pos
            for x, y in self.pixel_positions:
                u, v = self.convertor.xy_to_uv(x, y)
                theta_p, phi_p = self.convertor.uv_to_theta_phi(u, v, phi_c, theta_c, f)
                u_reconstructed, v_reconstructed = self.convertor.theta_phi_to_uv(theta_c, phi_c, f, theta_p, phi_p)
                self.assertEqual(u, u_reconstructed)
                self.assertEqual(v, v_reconstructed)


class TestSunGeometry(unittest.TestCase):
    def test_sun_calibration_on_ceske_budejovice(self):
        sun_thetas = np.deg2rad(
            np.round(np.array([70.0946364332799, 85.7480301427796, 83.39623639591332, 88.01104850711113]), 0))
        sun_phis = np.deg2rad(
            np.round(np.array([256.31033095068807, 283.93489695612186, 281.1575399273908, 286.7288952735205]), 0))
        xs = np.array([400, 1184, 1113, 1280])
        ys = np.array([137, 586, 508, 650])

        theta_calib, phi_calib, f_calib = calibrate(sun_thetas, sun_phis, xs, ys)
        expected_theta = np.deg2rad(86.1167975439323)
        expected_phi = np.deg2rad(270.29897608318)
        expected_f = 1579.6393131564218

        self.assertAlmostEqual(theta_calib, expected_theta, places=6)
        self.assertAlmostEqual(phi_calib, expected_phi, places=6)
        self.assertAlmostEqual(f_calib, expected_f, places=6)


if __name__ == '__main__':
    unittest.main()
