#include <iostream>
#include <vector>
//#include <omp.h>
#include "ArPragueSkyModelGroundXYZ/ArPragueSkyModelGroundXYZ.h"
#include "ArPragueSkyModelGroundXYZ/ArPragueSkyModelGroundXYZ.c"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>


#define SKYMODEL_DATA_FILE "ArPragueSkyModelGroundXYZ/SkyModelDataset.dat"

namespace py = pybind11;

double sun_elevation_ = 0;
double visibility_ = 0;
double albedo_ = 0;
ArPragueSkyModelGroundState * state = arpragueskymodelground_state_alloc_init(SKYMODEL_DATA_FILE, sun_elevation_, visibility_, albedo_);

std::vector<double> batch_luminance(double sun_elevation, double visibility, double albedo, std::vector<double> &thetas, std::vector<double> &gammas) {
    //auto start_time = std::chrono::high_resolution_clock::now();
    if (! (sun_elevation == sun_elevation_ && visibility == visibility_ && albedo == albedo_)) {
        state->elevation = sun_elevation;
        state->albedo = albedo;
        state->visibility = visibility;
        //state = arpragueskymodelground_state_alloc_init(SKYMODEL_DATA_FILE, sun_elevation, visibility, albedo);
        sun_elevation_ = sun_elevation;
        visibility_ = visibility;
        albedo_ = albedo;
    }

    //auto end_time = std::chrono::high_resolution_clock::now();
    //auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();   
    //std::cout << "Time taken: " << elapsed_time << " milliseconds" << std::endl;

    int length = thetas.size();
    std::vector<double> luminances(length, 0.0);

    
    //#pragma omp parallel for
    for (int i=0; i<length; i++) {
        if (thetas[i] >= 1.57079632679) luminances[i] = 0;//std::cout<<"Mala theta " << thetas[i] << std::endl;
        else luminances[i] = arpragueskymodelground_sky_radiance(state, thetas[i], gammas[i], 0.0, 1);
    }
    
    return luminances;
}

PYBIND11_MODULE(sky_image_generator, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
	m.def("batch_luminance", &batch_luminance,
          "A function that takes sun elevation, sun azimuth, visibility, albedo, "
          "and two lists of thetas and gammas, and returns a list of luminances",
          py::arg("sun_elevation"),
          py::arg("visibility"),
          py::arg("albedo"),
          py::arg("thetas"),
          py::arg("gammas"));
}
