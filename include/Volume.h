#pragma once
#include <vector> 
#include <complex>
#include <fftw3.h>

namespace feature_enhancement {
  class FFT;

  enum Domain {
    Time,
    Frequency
  };

  class Volume {
    friend class FFT;
    friend class VolumeList;
  public:
    Volume(size_t width, size_t height, size_t depth);
    Volume(Volume &&tmp);

    Volume(Volume const &other) = delete;
    Volume() = delete;
    void operator=(Volume const &other) = delete;

    ~Volume();

    double& operator()(size_t x, size_t y, size_t z);
    const double& operator()(size_t x, size_t y, size_t z) const;

    Volume& operator*=(Volume const &rhs);
    Volume& operator*=(double const &rhs);
    Volume& operator*=(std::complex<double> const &rhs);

    const size_t width, height, depth, size;
  private:
    const size_t complex_depth, real_depth, complex_size, real_size;
    double *real_data;
    std::complex<double> *complex_data;
    const bool free;
    Domain domain;

    // Note that it is the responsibility of the caller to allocate and free the
    // memory pointed to by data. Remember to take padding into consideration.
    Volume(double *data, size_t width, size_t height, size_t depth);
  };

  inline Volume operator*(Volume lhs, Volume const &rhs) {
    lhs *= rhs;
    return lhs;
  }

  template <typename NumType>
  inline Volume operator*(Volume lhs, NumType const &rhs) {
    lhs *= rhs;
    return lhs;
  }



  // VolumeList ensures that the Volumes are allocated contigously
  class VolumeList {
    friend class FFT;
  public:
    VolumeList(size_t n, size_t width, size_t height, size_t depth);
    ~VolumeList();

    VolumeList() = delete;
    VolumeList(VolumeList const &other) = delete;
    void operator=(VolumeList const &other) = delete;

    Volume& operator[](size_t n);    
    const Volume& operator[](size_t n) const;
    size_t size();


  private:
    size_t volume_size_complex, volume_size_real;
    double *data;
    std::vector<Volume> volumes;

  };

}

