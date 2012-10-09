#pragma once
#include <math.h>
#include "Filter.h"
#include "pechin_wrap.h"

namespace feature_enhancement {
  class GaussFilter3D: public Filter {
  public:
    typedef double (GaussFilter3D::*FilterFunction)(double, double, double) const;
    GaussFilter3D();
    ~GaussFilter3D() {}

    std::vector<double> get_kernel(FilterFunction f, int scale);
    
    void apply_gauss(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::gauss);
    }

    void apply_dx(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dx);
    }
    void apply_dy(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dy);
    }
    void apply_dz(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dz);
    }

    void apply_dxx(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dxx);
    }
    void apply_dxy(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dxy);
    }
    void apply_dxz(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dxz);
    }

    void apply_dyy(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dyy);
    }
    void apply_dyz(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dyz);
    }
    void apply_dzz(cimg_library::CImg<short> &volume, int scale) {
      this->apply(volume, scale, &GaussFilter3D::dzz);
    }

    // The standard Gauss filter in 3D
    double gauss(double x, double y, double z) const {
      return 
	this->normalise * 
	this->scale_frac_pow3 * 
	exp((-(x*x) - (y*y) - (z*z) ) / this->scale_pow2_2);
    }

    // first order derivatives
    double dx(double x, double y, double z) const {
      return -x * this->scale_frac_pow2 * gauss(x, y, z);
    }

    double dy(double x, double y, double z) const {
      return -y * this->scale_frac_pow2 * gauss(x, y, z);
    }

    double dz(double x, double y, double z) const {
      return -z * this->scale_frac_pow2 * gauss(x, y, z);
    }

    // second order derivatives
    double dxx(double x, double y, double z) const {
      return (x*x - scale_pow2) * scale_frac_pow4 * gauss(x, y, z);
    }

    double dyy(double x, double y, double z) const {
      return (y*y - scale_pow2) * scale_frac_pow4 * gauss(x, y, z);
    }

    double dzz(double x, double y, double z) const {
      return (z*z - scale_pow2) * scale_frac_pow4 * gauss(x, y, z);
    }

    double dxy(double x, double y, double z) const {
      return x * y * scale_frac_pow4 * gauss(x, y, z);
    }

    double dxz(double x, double y, double z) const {
      return x * z * scale_frac_pow4 * gauss(x, y, z);
    }

    double dyz(double x, double y, double z) const {
      return y * z * scale_frac_pow4 * gauss(x, y, z);
    }

    void apply(cimg_library::CImg<short> &volume, int scale, FilterFunction f);
  private:
    void set_scale(int scale);

    double scale_pow2;
    double scale_pow2_2;
    double scale_frac;
    double scale_frac_pow2;
    double scale_frac_pow3;
    double scale_frac_pow4;
    
    double const normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
  };
}
