folder_path = '../data/images-matlab';

% List subfolders in specified folder (excluding '.' and '..')
subfolders = dir(fullfile(folder_path, '*'));
subfolders = subfolders([subfolders(:).isdir] & ~ismember({subfolders(:).name},{'.','..'}));

% Loop through each subfolder and run the function
fprintf("{\n")
for i = 1:length(subfolders)
    location = subfolders(i).name;
    
    % Apply function and save results
    [focalLength, zenithAngle] = demoSkyCalib(location);
    fprintf('    "%s": {"focalLength": %f, "zenithAngle": %f}', location, focalLength, zenithAngle*180/pi)
    if i < length(subfolders)
        fprintf(",\n")
    end
    %fprintf('%s\n',fullfile(folder_path, location, 'results.mat'))
    %save(fullfile(folder_path, location, 'results.mat'), 'focalLength', 'zenithAngle');
end
fprintf("\n}\n")