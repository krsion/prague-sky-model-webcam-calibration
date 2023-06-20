FROM python:3.10

RUN pip3 install numpy scipy Pillow matplotlib pytz timezonefinder pysolar opencv-python-headless pybind11

WORKDIR /usr/bakalarka

COPY ./data/images ./data/images
COPY ./data/sky-masks ./data/sky-masks

COPY ./src ./src


