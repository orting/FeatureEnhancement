#pragma once
#include <vector>
#include <unordered_map>
#include <complex>
#include <fftw3.h>
#include "Volume.h"

namespace feature_enhancement {
  class FFT {
  public:
    FFT(size_t threads=1);
    ~FFT();
    void forward(Volume &volume);
    void backward(Volume &volume);
    void forward(VolumeList &volumes);
    void backward(VolumeList &volumes);

  private:
    std::unordered_map<double *, fftw_plan> forward_plans, backward_plans;
    // void backward(VolumeList< std::complex<double> > &in, VolumeList<double> &out);
    // void backward(VolumeList2 &volumes);
  };
}
