
FROM ubuntu:latest 

RUN apt-get update && apt-get install -y g++ python3 python3-pip python3-dev

RUN pip install numpy==1.25 scipy==1.11 Pillow==10.0 matplotlib==3.7 pytz==2023.3 timezonefinder==6.2 pysolar==0.10 opencv-python-headless==4.8.0.74 pybind11==2.10 h5py==3.9 jsonlines==3.1

WORKDIR /usr/bakalarka

COPY . .

WORKDIR /usr/bakalarka/src

RUN c++ -O3 -fopenmp -shared -fPIC -std=c++17  -w $(python3-config --cflags --ldflags) $(python3 -m pybind11 --includes) sky_image_generator.cpp -o /usr/lib/python3/dist-packages/sky_image_generator$(python3-config --extension-suffix)
# -O3 ... maximum optimization
# -fopenmp ... enable OpenMP (for parallelization)
# -undefined dynamic_lookup ... MACOS ONLY ... allow undefined symbols (for pybind11)s
# -shared -fPIC ... create a shared library
# -std=c++17 ... use C++17
# -w ... suppress warnings
# $(python3.10-config --cflags --ldflags) ... get the flags for compiling and linking against Python
# $(python3.10 -m pybind11 --includes) ... get the flags for compiling against pybind11
# sky_image_generator.cpp ... the source file
# -o sky_image_generator$(python3-config --extension-suffix) ... the output file