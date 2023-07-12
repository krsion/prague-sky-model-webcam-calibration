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
                print(num_tries, xs_moved, ys_moved, 'calib theta, phi, f:',
                    np.rad2deg(theta_calib), np.rad2deg(phi_calib), f_calib)
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
    print(data_W, data_H, data)
    perspective_calib = PerspectiveCalibrator(data_W, data_H)

    def run():
        '''
        dukovany
        6 [ 861  323  574 1123] [587 338 246 260] calib theta, phi, f: 85.54215277584741 133.8201134816082 1347.7694985918786
        14 [ 881  224  607 1188] [592 350 250 320] calib theta, phi, f: 83.02246726162623 133.21878943458177 1629.7994308443053
        sun azimuths: [2.267056088022244, 2.12490054857925, 2.3245945256044376, 2.422893808162115]
        sun zeniths: [1.4785162926039894, 1.2503275853759186, 1.1373974285361186, 1.3833045891625673]
        theta, phi, f = 1.4710038110135613 2.330354043400501 1488.7844647180918

        brno
        2113 [ 927  598 1164 1139] [569 722 206 740] calib theta, phi, f: 66.7330671785782 74.80109469190099 1430.7842593886924
        2747 [1035  607 1276 1079] [501 669 141 691] calib theta, phi, f: 68.8713840605408 73.40914603387753 1485.6176997444297
        sun azimuths: [1.4052737269395708, 1.2211088973962558, 1.5976590861559747, 1.499605241051938]
        sun zeniths: [1.165834954955128, 1.3304438601582307, 0.9957745580366999, 1.0811099287833807]
        theta, phi, f = 1.1833720772413656 1.2933783429191128 1458.200979566561

        ceske_budejovice
        6652 [ 312 1154 1013 1338] [231 541 554 553] calib theta, phi, f: 86.18786240733078 271.45329201773365 1591.6790450778599
        31021 [ 461 1203 1159 1257] [188 597 466 704] calib theta, phi, f: 85.76562497582414 268.88647165621484 1503.7279881704678
        sun azimuths: [4.4734591819658345, 4.955598813195152, 4.907124788540388, 5.0043632831289395]
        sun zeniths: [1.223382193749109, 1.4965854530908473, 1.4555389088802169, 1.5360825745815292]
        theta, phi, f = 1.5005772575612908 4.715353977779227 1547.703516624164

        holesov
        102 [1577  571 1052  746] [1026   92  680  426] calib theta, phi, f: 85.07816800061103 266.72424435637583 1632.2052171777357
        134 [1608  394 1136  735] [918  60 634 368] calib theta, phi, f: 86.76839551265633 267.6011652998822 1770.6908055621245
        sun azimuths: [5.084267599273715, 4.479232089722661, 4.832398845040084, 4.633982391832488]
        sun zeniths: [1.7300240274744247, 1.225431481031609, 1.5191680834349042, 1.351478231859353]
        theta, phi, f = 1.4996441707720347 4.662868837784603 1701.44801136993

        nedvezi
        1822 [316 357 438 752] [696 596 516 386] calib theta, phi, f: 76.9329121316974 88.07619778462234 2295.5445674436473
        2105 [293 379 489 744] [774 606 481 395] calib theta, phi, f: 76.50205122491471 87.06135457723249 2525.4752913528055
        sun azimuths: [1.309303760712018, 1.3571474961042482, 1.4050404897807605, 1.4854429159777893]
        sun zeniths: [1.3920244758887297, 1.3510635971871707, 1.3095969620973107, 1.2396884688362921]
        theta, phi, f = 1.338972649124866 1.5283634662991692 2410.5099293982266

        polom
        42997 [648 514 484 470] [579 614 639 653] calib theta, phi, f: 83.33629747091729 145.96249677712134 1466.489137094524
        70629 [711 581 482 407] [709 620 633 651] calib theta, phi, f: 82.6188384101165 142.97260648753203 1939.115827586643
        sun azimuths: [5.559424264105352, 5.523354786871178, 5.487752264598693, 5.452603557542627]
        sun zeniths: [1.6998657115059579, 1.6810722371879987, 1.661555368741246, 1.641348993020949]
        theta, phi, f = 1.44823176585931 2.521434438279009 1702.8024823405835

        pribyslav
        36617 [1429  474  868 1109] [881 280 376 663] calib theta, phi, f: 84.63500653514733 262.1947301931946 1652.3498121012888
        54185 [1381  463  937 1154] [862 156 456 684] calib theta, phi, f: 84.90175491356933 262.0044550969768 1716.284879185455
        sun azimuths: [4.910783635861304, 4.401142241385243, 4.627347006763365, 4.760744097674366]
        sun zeniths: [1.6453963997729204, 1.2258201104437971, 1.4047386687280325, 1.5153572212331916]
        theta, phi, f = 1.4794873452241486 4.574500859792659 1684.317345643372

        primda
        133969 [370 378 579 589] [583 401 379 532] calib theta, phi, f: 77.08211552513792 78.11080652614042 3664.448765190466
        158026 [402 429 518 598] [595 488 417 431] calib theta, phi, f: 76.67485270749782 77.19588455065684 4340.4273150238605
        sun azimuths: [1.2495878529241653, 1.2651270306714595, 1.2806588231392722, 1.2961888250716342]
        sun zeniths: [1.3320651021620116, 1.318689735645354, 1.3052430912886237, 1.2917290954781495]
        theta, phi, f = 1.34178267177191 1.355306554833905 4002.438040107163

        olomouc
        17 [1041  786  440  625] [368 409 512 459] calib theta, phi, f: 81.4263140169528 162.98768047466885 1362.7833300047075
        471 [984 734 382 576] [393 533 581 584] calib theta, phi, f: 77.77271713662024 165.15105246850962 1380.59059224907
        sun azimuths: [3.014280202457452, 2.8472978249765823, 2.569201303521834, 2.7254708468770477]
        sun zeniths: [1.2598302465970161, 1.2835016555767351, 1.3664189021598747, 1.3129532715416707]
        theta, phi, f = 1.389273629807438 2.863550646590424 1371.6869611268887

        temelin
        868 [156 642 295 473] [419 118 410 312] calib theta, phi, f: 81.51051584296454 141.67457341441195 1783.327285278011
        3925 [252 629 409 521] [488 214 442 344] calib theta, phi, f: 81.44551814254501 142.39655575651727 1432.5725486185977
        sun azimuths: [2.105588849576309, 2.3748506972979984, 2.1979118493332037, 2.2943727935926193]
        sun zeniths: [1.3547028828875933, 1.1985708307555536, 1.295297368186787, 1.2397392236387879]
        theta, phi, f = 1.4220596645194596 2.4789882569454122 1607.9499169483042
        '''
        sun_calibrations = {}
        for location in data:
            if location in [ 'belotin', 'brno']:
                continue
            solar_data = [sun_calc.sun_position(f) for f in data[location]['filenames']]
            data[location]['sun_thetas'] = [x['sunZenith'] for x in solar_data]
            data[location]['sun_phis'] = [x['sunAzimuth'] for x in solar_data]
            theta, phi, f = perspective_calib.calibrate(data[location]['sun_thetas'], data[location]['sun_phis'],
                                      np.array(data[location]['xs']), np.array(data[location]['ys']))
            print(location, 'theta, phi, f =', theta, phi, f)
            sun_calibrations[location] = {}
            sun_calibrations[location]['theta'] = theta
            sun_calibrations[location]['phi'] = phi
            sun_calibrations[location]['f'] = f

        print(sun_calibrations)

    sun_calibrations = {'dukovany': {'theta': 1.4710038110135613, 'phi': 2.330354043400501, 'f': 1488.7844647180918},
                    'brno': {'theta': 1.1833720772413656, 'phi': 1.2933783429191128, 'f': 1458.200979566561},
                    'ceske_budejovice': {'theta': 1.5005772575612908, 'phi': 4.715353977779227, 'f': 1547.703516624164},
                    'holesov': {'theta': 1.4996441707720347, 'phi': 4.662868837784603, 'f': 1701.44801136993},
                    'nedvezi': {'theta': 1.338972649124866, 'phi': 1.5283634662991692, 'f': 2410.5099293982266},
                    'polom': {'theta': 1.44823176585931, 'phi': 2.521434438279009, 'f': 1702.8024823405835},
                    'pribyslav': {'theta': 1.4794873452241486, 'phi': 4.574500859792659, 'f': 1684.317345643372},
                    'primda': {'theta': 1.34178267177191, 'phi': 1.355306554833905, 'f': 4002.438040107163},
                    'olomouc': {'theta': 1.389273629807438, 'phi': 2.863550646590424, 'f': 1371.6869611268887},
                    'temelin': {'theta': 1.4220596645194596, 'phi': 2.4789882569454122, 'f': 1607.9499169483042}}
    run()
