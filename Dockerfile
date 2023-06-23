FROM python:3.10

RUN pip3 install numpy scipy Pillow matplotlib pytz timezonefinder pysolar opencv-python-headless pybind11 h5py

WORKDIR /usr/bakalarka

COPY . .