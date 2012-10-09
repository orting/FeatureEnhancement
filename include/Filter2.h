#include <complex>
#include <fftw3.h>
#include "pechin_wrap.h"

namespace feature_enhancement {
  namespace {
    template<typename NumType>  
    void center_fft(cimg_library::CImg<NumType> &vol, int w, int h, int d) {
      d = std::min(d, vol.depth());
      h = std::min(h, vol.height());
      w = std::min(w, vol.width());
      double tmp;
      for (int z = 0; z < d; ++z) {
	for (int y = 0; y < h; ++y) {
	  for (int x = 0, xx = w/2; xx < w; ++x, ++xx) {
	    tmp = vol(x,y,z);
	    vol(x,y,z) = vol(xx,y,z);
	    vol(xx,y,z) = tmp;
	  }
	}
      }
      for (int z = 0; z < d; ++z) {
	for (int y = 0, yy = h/2; yy < h; ++y, ++yy) {
	  for (int x = 0; x < w; ++x) {
	    tmp = vol(x,y,z);
	    vol(x,y,z) = vol(x,yy,z);
	    vol(x,yy,z) = tmp;
	  }
	}
      }
      for (int z = 0, zz = d/2; zz < d; ++z, ++zz) {
	for (int y = 0; y < h; ++y) {
	  for (int x = 0; x < w; ++x) {
	    tmp = vol(x,y,z);
	    vol(x,y,z) = vol(x,y,zz);
	    vol(x,y,zz) = tmp;
	  }
	}
      }
    }

    template<typename NumType1, typename NumType2>
    void normalise_and_copy(cimg_library::CImg<NumType1> const &src,
			    cimg_library::CImg<NumType2> &dst) {
      double scale = dst.width() * dst.height() * dst.depth();
      cimg_forXYZ(dst, x, y, z) {
	dst(x, y, z) = static_cast<NumType2>(src(x,y,z) / scale);	
      }
    }
  }

  template<typename NumType>
  void apply(cimg_library::CImg<NumType> &volume, 
	     int scale, 
	     std::function<NumType(NumType, NumType, NumType, int)> filter_function) {

    int w = volume.width();
    int h = volume.height();
    int d = volume.depth();

    cimg_library::CImg<double> complex_volume(2*(w/2 + 1), h, d);
    cimg_library::CImg<double> filter(2*(w/2 + 1), h, d);

    // transform the filter
    double *fft_filter_real = filter.data();
    fftw_complex *fft_filter_complex = (fftw_complex *) filter.data();
    fftw_plan filter_forward = fftw_plan_dft_r2c_3d(d, h, w, fft_filter_real, fft_filter_complex, FFTW_ESTIMATE);

    // Transform the volume
    double *fft_volume_real = complex_volume.data();
    fftw_complex *fft_volume_complex = (fftw_complex *) complex_volume.data();
    fftw_plan volume_forward = fftw_plan_dft_r2c_3d(d, h, w, fft_volume_real, fft_volume_complex, FFTW_ESTIMATE);
    fftw_plan volume_backward = fftw_plan_dft_c2r_3d(d, h, w, fft_volume_complex, fft_volume_real, FFTW_ESTIMATE);

    cimg_forXYZ(volume, x, y, z) {
      complex_volume(x,y,z) = volume(x,y,z);
    }


    int wc = w/2;
    int hc = h/2;
    int dc = d/2;
    int window = 20 * scale; // We do ± 3 standard deviations ??
    int start_x = wc - window > 0 ? - window : - wc;
    int start_y = hc - window > 0 ? - window : - hc;
    int start_z = dc - window > 0 ? - window : - dc;
    int end_x = wc + window < w ? window : wc;
    int end_y = hc + window < h ? window : hc;
    int end_z = dc + window < d ? window : dc;

    for (int x = start_x, i = wc + start_x; x <= end_x; ++x, ++i) {
      for (int y = start_y, j = hc + start_y; y <= end_y; ++y, ++j) {
	for (int z = start_z, k = dc + start_z; z <= end_z; ++z, ++k) {
	  filter(i, j, k) = filter_function(x, y, z, scale);
	}
      }
    }

    fftw_execute(filter_forward);
    fftw_execute(volume_forward);

    fftw_destroy_plan(filter_forward);
    fftw_destroy_plan(volume_forward);

    int size = (filter.width()/2) * filter.height() * filter.depth();
    std::complex<double> *vol = (std::complex<double>*) fft_volume_complex;
    std::complex<double> *fil = (std::complex<double>*) fft_filter_complex;
    for (int i = 0; i < size; ++i) {
      vol[i] *= fil[i];
    }

    fftw_execute(volume_backward);
    center_fft(complex_volume, w, h, d);
    normalise_and_copy(complex_volume, volume);
  }

}
