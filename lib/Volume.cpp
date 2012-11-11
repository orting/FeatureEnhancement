#include <complex>
#include <fftw3.h>
#include <cstring>
#include "Volume.h"
#include <iostream>

namespace feature_enhancement {  
  Volume::Volume(size_t width, size_t height, size_t depth)
    : width(width), 
      height(height), 
      depth(depth),
      size(width * height * depth),
      complex_depth(depth/2 + 1),
      real_depth(2 * complex_depth),
      complex_size(width * height * complex_depth),
      real_size(2 * complex_size),
      real_data(fftw_alloc_real(real_size)),
      complex_data(reinterpret_cast< std::complex<double>* >(real_data)),
      free(true),
      domain(Domain::Time)
  {
    std::memset(real_data, 0, sizeof(double) * real_size);
  }
  
  Volume::Volume(double *data, size_t width, size_t height, size_t depth)
    : width(width), 
      height(height), 
      depth(depth),
      size(width * height * depth),
      complex_depth(depth/2 + 1),
      real_depth(2 * complex_depth),
      complex_size(width * height * complex_depth),
      real_size(2 * complex_size),
      real_data(data),
      complex_data(reinterpret_cast< std::complex<double>* >(data)),
      free(false),
      domain(Domain::Time)
  {}

  Volume::Volume(Volume const &other)
    : Volume(other.width, other.height, other.depth)
  {
    std::copy(other.real_data, other.real_data + real_size, real_data);
    domain = other.domain;
  }

  Volume::Volume(Volume &&other)
    : Volume(other.real_data, other.width, other.height, other.depth)
  {}


  Volume::~Volume() {
    if (free) {
      fftw_free(real_data);
    }
  }

  double& Volume::operator()(size_t x, size_t y, size_t z) {
    return real_data[x * real_depth * height + y * real_depth + z];
  }

  const double& Volume::operator()(size_t x, size_t y, size_t z) const {
    return real_data[x * real_depth * height + y * real_depth + z];
  }

  Volume& Volume::operator*=(Volume const &rhs) {
    if (real_size == rhs.real_size && domain == rhs.domain) {
      if (domain == Domain::Time) {
	for (size_t x = 0; x < width; ++x) {
	  for (size_t y = 0; y < height; ++y) {
	    for (size_t z = 0; z < depth; ++z) {
	      operator()(x, y, z) *= rhs(x, y, z);
	    }
	  }
	}
      }
      else {
	for (std::complex<double> *p1 = complex_data, *p2 = rhs.complex_data;
	     p1 < complex_data + complex_size;
	     ++p1, ++p2) {
	  *p1 *= *p2;
	}
      }
    }
    return *this;
  }

  Volume& Volume::operator*=(double const &rhs) {
    for (size_t x = 0; x < width; ++x) {
      for (size_t y = 0; y < height; ++y) {
	for (size_t z = 0; z < depth; ++z) {
	  (*this)(x, y, z) *= rhs;
	}
      }
    }
    return *this;
  }

  Volume& Volume::operator*=(std::complex<double> const &rhs) {
    for (std::complex<double> *p1 = complex_data; p1 < complex_data + complex_size; ++p1) {
      *p1 *= rhs;
    }
    return *this;
  }  
}
