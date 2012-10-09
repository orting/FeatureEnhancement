#include <iostream>
#include <math.h>
#include "Filter.h"



void Filter::apply(cimg_library::CImg<double> &volume, int w, int h, int d, int scale) {
  cimg_library::Cimg<double> filter(2*(w/2 +1), h, d, 1, 0);
  double *in = filter.data();
  fftw_complex *out = (fftw_complex *) filter.data();
  fftw_plan forward = fftw_plan_dft_r2c_3d(d, h, w, in, out, FFTW_ESTIMATE);

  int wc = w/2;
  int hc = h/2;
  int dc = c/2;
  int window = 3 * scale; // We do ± 3 standard deviations
  int start_x = wc - window > 0 ? - window : - wc;
  int start_y = hc - window > 0 ? - window : - hc;
  int start_z = dc - window > 0 ? - window : - dc;
  int end_x = wc + window < w ? window : wc;
  int end_y = hc + window < h ? window : hc;
  int end_z = dc + window < d ? window : dc;

  for (int x = start_x, i = wc + start_x; x <= end_x; ++x, ++i) {
    for (int y = start_y, j = hc + start_y; y <= end_y; ++y, ++j) {
      for (int z = start_z, k = dc + start_z; z <= end_z; ++z, ++k) {
	filter(i, j, k) = this->current_filter(x,y,z);
      }
    }
  }
    
  fftw_execute(forward);

  cimg_forXYZ(volume, x, y, z) {
    volume(x,y,z) *= filter(x,y,z);
  }

  fftw_destroy_plan(forward);
}


// Create the gaussfilter, shold perhaps be saved and loaded ?
static double const normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
double const k = normalise * (1.0 / (scale * scale * scale));
double const s2 = 2 * scale * scale;
cimg_library::CImg<double> filter



return 
this->normalise * 
this->scale_frac_pow3 * 
exp((-(x*x) - (y*y) - (z*z) ) / this->scale_pow2_2);  
}
void apply_dx_freq(cimg_library::CImg<double> volume, int scale);
void apply_dy_freq(cimg_library::CImg<double> volume, int scale);
void apply_dz_freq(cimg_library::CImg<double> volume, int scale);
void apply_dxx_freq(cimg_library::CImg<double> volume, int scale);
void apply_dxy_freq(cimg_library::CImg<double> volume, int scale);
void apply_dxz_freq(cimg_library::CImg<double> volume, int scale);
void apply_dyy_freq(cimg_library::CImg<double> volume, int scale);
void apply_dyz_freq(cimg_library::CImg<double> volume, int scale);
void apply_dzz_freq(cimg_library::CImg<double> volume, int scale);
