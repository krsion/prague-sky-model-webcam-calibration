function chmuCalibration(iPath, jPath)

locations = {'brno', 'ceske_budejovice', 'dukovany', 'holesov', 'nedvezi', 'olomouc', 'polom', 'pribyslav', 'primda', 'temelin'};

fprintf("{\n")
for i = 1:length(locations)
    location = locations{i};

    imagesPath = '../data-matlab';
    gradientPath = strcat( iPath, '/', location);
    clearDayPath = strcat( jPath, '/', location);
    skyMaskPath = strcat('../data-matlab/masks/', location, '.jpg');

    [focalLength, zenithAngle, azimuthAngle] = calibrate(imagesPath, gradientPath, clearDayPath, skyMaskPath);
    fprintf('"%s": {"f": %.2f , "theta": %.2f , "phi": %.2f },\n', location, focalLength, zenithAngle*180/pi, azimuthAngle*180/pi);
end
fprintf("\n}\n")