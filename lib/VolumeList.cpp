#include <iostream>
#include <complex>
#include <fftw3.h>
#include <cstring>
#include "VolumeList.h"

namespace feature_enhancement {
  VolumeList::VolumeList(size_t n, Volume const &vol)
    : VolumeList(n, vol.width, vol.height, vol.depth)
  {}
    
  VolumeList::VolumeList(size_t n, size_t width, size_t height, size_t depth)
    : volume_size_complex(width * height * (depth/2 + 1)),
      volume_size_real(2 * volume_size_complex),
      data(fftw_alloc_real(n * volume_size_real)),
      volumes()
  {
    std::memset(data, 0, sizeof(double) * n * volume_size_real);
    volumes.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      volumes.push_back(Volume(data + i * volume_size_real, width, height, depth));
    }
  }

  VolumeList::~VolumeList() {
    fftw_free(data);
  }

  Volume& VolumeList::operator[](size_t n) {
    return volumes[n];
  }
    
  const Volume& VolumeList::operator[](size_t n) const {
    return volumes[n];
  }

  VolumeList& VolumeList::operator*=(Volume const &rhs) {
    for (size_t i = 0; i < size(); ++i) {
      volumes[i] *= rhs;
    }
    return *this;
  }

  size_t VolumeList::size() {
    return volumes.size();
  }
}
