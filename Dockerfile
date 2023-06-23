
FROM ubuntu:latest as builder

RUN apt-get update && apt-get install -y g++ python3 python3-pip python3-dev

RUN pip install numpy scipy Pillow matplotlib pytz timezonefinder pysolar opencv-python-headless pybind11 h5py

WORKDIR /usr/bakalarka

COPY . .

WORKDIR /usr/bakalarka/src

RUN c++ -O3 -shared -fPIC -std=c++17  -w $(python3-config --cflags --ldflags) $(python3 -m pybind11 --includes) sky_image_generator.cpp -o /usr/lib/python3/dist-packages/sky_image_generator$(python3-config --extension-suffix)
# -O3 ... maximum optimization
# -undefined dynamic_lookup ... MACOS ONLY ... allow undefined symbols (for pybind11)s
# -shared -fPIC ... create a shared library
# -std=c++17 ... use C++17
# -w ... suppress warnings
# $(python3.10-config --cflags --ldflags) ... get the flags for compiling and linking against Python
# $(python3.10 -m pybind11 --includes) ... get the flags for compiling against pybind11
# sky_image_generator.cpp ... the source file
# -o sky_image_generator$(python3-config --extension-suffix) ... the output file