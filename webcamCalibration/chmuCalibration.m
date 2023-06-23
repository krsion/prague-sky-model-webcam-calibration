
locations = {'brno', 'ceske_budejovice', 'dukovany', 'holesov', 'nedvezi', 'olomouc', 'polom', 'pribyslav', 'primda', 'temelin'};

fprintf("{\n")
for i = 1:length(locations)
    location = locations{i};

    imagesPath = '../data-matlab';
    gradientPath = strcat('I/', location);
    clearDayPath = strcat('J/', location);
    skyMaskPath = strcat('../data-matlab/sky-masks/', location, '.jpg');

    [focalLength, zenithAngle, azimuthAngle] = calibrate(imagesPath, gradientPath, clearDayPath, skyMaskPath);
    fprintf('%s: Estimated focal length: %.2f px, zenith angle: %.2f deg, azimuth angle: %.2f deg\n', location, focalLength, zenithAngle*180/pi, azimuthAngle*180/pi);
end
fprintf("\n}\n")