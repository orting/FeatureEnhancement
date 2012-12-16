#pragma once
#include <vector> 
#include <complex>
#include <fftw3.h>
#include "Volume.h"

namespace feature_enhancement {
  class FFT;
  // VolumeList ensures that the Volumes are allocated contigously
  class VolumeList {
    friend class FFT;
  public:
    VolumeList(size_t n, Volume const &vol);
    VolumeList(size_t n, size_t width, size_t height, size_t depth);
    ~VolumeList();

    VolumeList() = delete;
    VolumeList(VolumeList const &other) = delete;
    void operator=(VolumeList const &other) = delete;

    Volume& operator[](size_t n);    
    const Volume& operator[](size_t n) const;

    VolumeList& operator*=(Volume const &rhs);
    //    Volume& operator*=(double const &rhs);
    //    Volume& operator*=(std::complex<double> const &rhs);

    size_t size();


  private:
    size_t volume_size_complex, volume_size_real;
    double *data;
    std::vector<Volume> volumes;

  };

  // inline VolumeList operator*(VolumeList lhs, Volume const &rhs) {
  //   lhs *= rhs;
  //   return lhs;
  // }

}

