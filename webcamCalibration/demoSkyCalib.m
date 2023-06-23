imagesPath = './images';
gradientPath = 'gradient';
clearDayPath = 'clearDay';
skyMaskPath = './skyMask/mask.jpg';

[focalLength, zenithAngle, azimuthAngle] = calibrate(imagesPath, gradientPath, clearDayPath, skyMaskPath);

fprintf('Estimated focal length: %.2f px, zenith angle: %.2f deg, azimuth angle: %.2f deg\n', focalLength, zenithAngle*180/pi, azimuthAngle*180/pi);