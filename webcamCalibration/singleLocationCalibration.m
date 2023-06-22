function [focalLength, zenithAngle] = singleLocationCalibration(location)   

imagesPath = 'data';
gradientPath = fullfile('I-train-matlab', location);
% clearDayPath = fullfile('J', location);
skyMaskPath = fullfile('../data/maskmorpho', strcat(location, '.jpg'));

gradientFileList = dir(fullfile(imagesPath, gradientPath, '*.jpg'));
gradientFileList = {gradientFileList.name};


nbRandomPixelsToKeep = 1000;

imgWidth = 1600;
imgHeight = 1200;

skyMask = im2double(imread(skyMaskPath))>0.5;



[xRange, yRange] = meshgrid(1:imgWidth, 1:imgHeight);
xpVec = (xRange - imgWidth/2) - 0.5;
ypVec = (imgHeight/2 - yRange) + 0.5;

xp = cell(1, length(gradientFileList));
yp = cell(1, length(gradientFileList));
lp = cell(1, length(gradientFileList));
    
for f=1:length(gradientFileList)
    [xp{f}, yp{f}, lp{f}] = loadImageInformation(fullfile(imagesPath, gradientPath, gradientFileList{f}), [], skyMask, xpVec, ypVec, nbRandomPixelsToKeep);
end

% initialize horizon to lowest sky row
[rSky, ~] = find(skyMask);
yh0 = ypVec(max(rSky(:)), 1) - 1;
f0 = 3000;
theta0 = pi/2+atan2(yh0, f0);

[focalLength, zenithAngle] = fitFocalAndZenith(xp, yp, lp', theta0, f0);

%{
clearDayFileList = dir(fullfile(imagesPath, clearDayPath, '*.jpg'));
clearDayFileList = {clearDayFileList.name};
xp = cell(1, length(clearDayFileList));
yp = cell(1, length(clearDayFileList));
lp = cell(1, length(clearDayFileList));
sunAzimuths = cell(1, length(clearDayFileList));
sunZeniths = cell(1, length(clearDayFileList));
for f=1:length(clearDayFileList)
    [xp{f}, yp{f}, lp{f}] = loadImageInformation(fullfile(imagesPath, clearDayPath, clearDayFileList{f}), [], skyMask, xpVec, ypVec, nbRandomPixelsToKeep);
    % load sun position
    load(fullfile(imagesPath, clearDayPath, strrep(clearDayFileList{f}, '.jpg', '.mat')), 'sunZenith', 'sunAzimuth');
    sunZeniths{f} = sunZenith;
    sunAzimuths{f} = sunAzimuth;
end
azimuthAngle = fitAzimuth(xp, yp, lp, focalLength, zenithAngle, sunAzimuths, sunZeniths); 

%}