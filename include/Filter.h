#pragma once
#include <complex>
#include <fftw3.h>
#include "pechin_wrap.h"

namespace filter {
  //
  // Kernels
  //
  // 1D Kernel
  template <typename NumType>
  std::vector<NumType> 
  kernel_1d(NumType (&filter_function)(NumType, int), int scale) {
    std::vector<NumType> kernel;
    int width = 2 * 3 * scale + 1;
    kernel.reserve(width);

    for (NumType x = -3*scale; x <= 3*scale; x += 1) {
      kernel.push_back(filter_function(x, scale));
    }
    return kernel;
  }

  // 2D Kernel
  template <typename NumType>
  std::vector< std::vector<NumType> > 
  kernel_2d(NumType (&filter_function)(NumType, NumType, int), int scale) {
    int width = 2 * 3 * scale + 1;

    std::vector< std::vector<NumType> > kernel;
    std::vector<NumType> ys;
    kernel.reserve(width);
    ys.resize(width);

    for (NumType x = -3*scale; x <= 3*scale; x += 1) {
      int i = 0;
      for (NumType y = -3*scale; y <= 3*scale; y += 1) {
	ys[i++] = filter_function(x, y, scale);
      }
      kernel.push_back(ys);
    }

    return kernel;
  }

  
  // 3D Kernel
  template <typename NumType>
  std::vector< std::vector< std::vector<NumType> > > 
  kernel_3d(NumType (&filter_function)(NumType, NumType, NumType, int),  int scale) {
    int width = 2 * 3 * scale + 1;

    std::vector< std::vector< std::vector<NumType> > > kernel;
    std::vector< std::vector<NumType> > ys;
    std::vector<NumType> zs;
    kernel.reserve(width);
    ys.resize(width);
    zs.resize(width);

    for (NumType x = -3*scale; x <= 3*scale; x += 1) {
      int i = 0;
      for (NumType y = -3*scale; y <= 3*scale; y += 1) {
	int j = 0;
	for (NumType z = -3*scale; z <= 3*scale; z += 1) {
	  zs[j++] = filter_function(x, y, z, scale);
	}
	ys[i++] = zs;
      }
      kernel.push_back(ys);
    }

    return kernel;
  }

  template <typename NumType>
  void kernel_3d(NumType (&filter_function)(NumType, NumType, NumType, int), 
		 int scale,
		 cimg_library::CImg<NumType> &out,
		 size_t out_w, size_t out_h, size_t out_d) {
    
    int half_w = out_w / 2;
    int half_h = out_h / 2;
    int half_d = out_d / 2;
    int kernel_size = 3 * scale;

    int i =  kernel_size > half_w ? 0 : half_w - kernel_size;
    for (NumType x = - kernel_size; x <= kernel_size; x += 1, ++i) {
      int j =  kernel_size > half_h ? 0 : half_h - kernel_size;
      for (NumType y = -kernel_size; y <= kernel_size; y += 1, ++j) {
	int k =  kernel_size > half_d ? 0 : half_d - kernel_size;
	for (NumType z = -kernel_size; z <= kernel_size; z += 1, ++k) {
	  out(i, j, k) = filter_function(x, y, z, scale);
	}
      }
    }
  }
  

  // Helper function for apply
  namespace {
    template<typename NumType1, typename NumType2>
    void normalise_and_copy(cimg_library::CImg<NumType1> const &src,
			    cimg_library::CImg<NumType2> &dst) {
      int w = dst.width();
      int h = dst.height();
      int d = dst.depth();
      int half_w = w % 2 ? w/2 + 1: w/2;
      int half_h = h % 2 ? h/2 + 1: h/2;
      int half_d = d % 2 ? d/2 + 1: d/2;
      double scale = w * h * d;
      cimg_forXYZ(dst, x, y, z) {
	int xx = (x + half_w) % w;
	int yy = (y + half_h) % h;
	int zz = (z + half_d) % d;
	dst(x, y, z) = static_cast<NumType2>(src(xx,yy,zz) / scale);	
      }
    }
  }


  // Apply the filter given by the filter_function to the volume
  template<typename CImgType, typename NumType>
  void apply(cimg_library::CImg<CImgType> &volume, 
	     int scale, 
	     NumType (&filter_function)(NumType, NumType, NumType, int)) {

    unsigned int w = volume.width();
    unsigned int padded_w = 2 * (w / 2 + 1);
    unsigned int h = volume.height();
    unsigned int d = volume.depth();

    // CImg uses column-major order, fftw exspects row-major, 
    // so the 1 dimension of CImg is padded as it is passed as the last to fftw
    cimg_library::CImg<double> padded(padded_w, h, d);
    cimg_library::CImg<double> filter(padded_w, h, d);

    // transform the filter
    double *filter_in = filter.data();
    fftw_complex *filter_out = reinterpret_cast<fftw_complex *>(filter.data());
    fftw_plan filter_forward = fftw_plan_dft_r2c_3d(d, h, w, filter_in, filter_out, FFTW_ESTIMATE);

    kernel_3d(filter_function, scale, filter, w, h, d);

    fftw_execute(filter_forward);
    fftw_destroy_plan(filter_forward);


    // Transform the volume
    double *volume_in = padded.data();
    fftw_complex *volume_out = reinterpret_cast<fftw_complex *>(padded.data());
    fftw_plan volume_forward = fftw_plan_dft_r2c_3d(d, h, w, volume_in, volume_out, FFTW_ESTIMATE);
    fftw_plan volume_backward = fftw_plan_dft_c2r_3d(d, h, w, volume_out, volume_in, FFTW_ESTIMATE);

    cimg_forXYZ(volume, x, y, z) {
      padded(x,y,z) = volume(x,y,z);
    }

    fftw_execute(volume_forward);
    fftw_destroy_plan(volume_forward);

    // Convolve
    int size = (padded_w * h * d) / (sizeof(std::complex<double>) / sizeof(double));
    std::complex<double> *complex_volume = reinterpret_cast<std::complex<double>*>(volume_out);
    std::complex<double> *complex_filter = reinterpret_cast<std::complex<double>*>(filter_out);
    int i = 0;
    for (i = 0; i < size; ++i) {
      complex_volume[i] *= complex_filter[i];
    }
    
    // copy back to volume
    fftw_execute(volume_backward);
    fftw_destroy_plan(volume_backward);
    normalise_and_copy(padded, volume);
  }

}
