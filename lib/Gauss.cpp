#include <vector>
#include <math.h>

namespace feature_enhancement {
  //
  // 1D 
  //
  // double gauss(double x, int scale) {
  //   static double const const_normalise = 0.3989422804014327; // = 1 / sqrt(2*pi)
  //   return (const_normalise / scale) * exp((-(x*x)) / (2 * scale * scale));
  // }

  // // 1D first order derivatives
  // double dx(double x, int scale) {
  //   static double const const_normalise = 0.3989422804014327; // = 1 / sqrt(2*pi)
  //   double scale_squared = scale * scale;
  //   double scale_cubed = scale_squared * scale;
  //   return 
  //     -x * 
  //     (const_normalise / scale_cubed) * 
  //     exp((-(x*x)) / (2 * scale_squared));
  // }


  // //
  // // 2D
  // //
  // double gauss(double x, double y, int scale) {
  //   static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
  //   double scale_squared = scale * scale;
  //   return (const_normalise / scale_squared) * exp((-(x*x) -(y*y)) / (2 * scale_squared));
  // }

  // // 2D first order gaussian derivatives
  // double dx(double x, double y, int scale) {
  //   static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
  //   double scale_squared = scale * scale;
  //   double scale_fourth = scale_squared * scale_squared;
  //   return 
  //     -x * 
  //     (const_normalise / scale_fourth) * 
  //     exp((-(x*x) - (y*y)) / (2 * scale_squared));
  // }

  // double dy(double x, double y, int scale) {
  //   return dx(y, x, scale);
  // }

  // // 2D second order gaussian derivatives
  // double dxx(double x, double y, int scale) {
  //   static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
  //   double scale_squared = scale * scale;
  //   double scale_sixth = scale_squared * scale_squared * scale_squared;
  //   return 
  //     (x*x - scale_squared) * 
  //     (const_normalise / scale_sixth) * 
  //     exp((-(x*x) - (y*y)) / (2 * scale_squared));
  // }

  // double dyy(double x, double y, int scale) {
  //   return dxx(y, x, scale);
  // }

  // double dxy(double x, double y, int scale) {
  //   static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
  //   double scale_squared = scale * scale;
  //   double scale_sixth = scale_squared * scale_squared * scale_squared;
  //   return 
  //     x * y *
  //     (const_normalise / scale_sixth) * 
  //     exp((-(x*x) - (y*y)) / (2 * scale_squared));
  // }

  
  //
  // 3D
  //
  double gauss3D(double x, double y, double z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_cubed = scale_squared * scale;
    return (const_normalise / scale_cubed) * exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  // 3D first order derivatives
  double dx3D(double x, double y, double z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_fifth = scale_squared * scale_squared * scale;
    return 
      -x * 
      (const_normalise / scale_fifth) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  double dy3D(double x, double y, double z, int scale) {
    return dx3D(y, x, z, scale);
  }

  double dz3D(double x, double y, double z, int scale) {
    return dx3D(z, y, x, scale);
  }

  // 3D second order derivatives
  double dxx3D(double x, double y, double z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_seventh = scale_squared * scale_squared * scale_squared * scale;
    return 
      (x*x - scale_squared) * 
      (const_normalise / scale_seventh) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  double dyy3D(double x, double y, double z, int scale) {
    return dxx3D(y, x, z, scale);
  }

  double dzz3D(double x, double y, double z, int scale) {
    return dxx3D(z, y, x, scale);
  }

  double dxy3D(double x, double y, double z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_seventh = scale_squared * scale_squared * scale_squared * scale;
    return 
      x * y *
      (const_normalise / scale_seventh) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  double dxz3D(double x, double y, double z, int scale) {
    return dxy3D(x, z, y, scale);
  }

  double dyz3D(double x, double y, double z, int scale) {
    return dxy3D(y, z, x, scale);
  }
}
