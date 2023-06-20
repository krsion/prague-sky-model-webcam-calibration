#!/bin/bash

echo "Copying dataset..."
cd ./src && python pre_matlab.py
echo "Running matlab..."
cd ../webcamCalibration
matlab -batch "chmuCalibration"
rm -r ../data/images-matlab
echo "Done!"
