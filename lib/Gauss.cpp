#include <vector>
#include <math.h>

namespace gauss {
  // 1D gaussian
  double gauss(double x, int scale) {
    static double const const_normalise = 0.3989422804014327; // = 1 / sqrt(2*pi)
    return (const_normalise / scale) * exp((-(x*x)) / (2 * scale * scale));
  }

  // 1D first order gaussian derivatives
  double dx(double x, int scale) {
    static double const const_normalise = 0.3989422804014327; // = 1 / sqrt(2*pi)
    double scale_squared = scale * scale;
    double scale_cubed = scale_squared * scale;
    return 
      -x * 
      (const_normalise / scale_cubed) * 
      exp((-(x*x)) / (2 * scale_squared));
  }


  // 2D gaussian
  double gauss(double x, double y, int scale) {
    static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
    double scale_squared = scale * scale;
    return (const_normalise / scale_squared) * exp((-(x*x) -(y*y)) / (2 * scale_squared));
  }

  // 2D first order gaussian derivatives
  double dx(double x, double y, int scale) {
    static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
    double scale_squared = scale * scale;
    double scale_fourth = scale_squared * scale_squared;
    return 
      -x * 
      (const_normalise / scale_fourth) * 
      exp((-(x*x) - (y*y)) / (2 * scale_squared));
  }

  double dy(double x, double y, int scale) {
    return dx(y, x, scale);
  }

  // 2D second order gaussian derivatives
  double dxx(double x, double y, int scale) {
    static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
    double scale_squared = scale * scale;
    double scale_sixth = scale_squared * scale_squared * scale_squared;
    return 
      (x*x - scale_squared) * 
      (const_normalise / scale_sixth) * 
      exp((-(x*x) - (y*y)) / (2 * scale_squared));
  }

  double dyy(double x, double y, int scale) {
    return dxx(y, x, scale);
  }

  double dxy(double x, double y, int scale) {
    static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
    double scale_squared = scale * scale;
    double scale_sixth = scale_squared * scale_squared * scale_squared;
    return 
      x * y *
      (const_normalise / scale_sixth) * 
      exp((-(x*x) - (y*y)) / (2 * scale_squared));
  }


  // 3D gaussian
  double gauss(double x, double y, double z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_cubed = scale_squared * scale;
    return (const_normalise / scale_cubed) * exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  // 3D first order gaussians derivatives
  double dx(double x, double y, double z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_fifth = scale_squared * scale_squared * scale;
    return 
      -x * 
      (const_normalise / scale_fifth) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  double dy(double x, double y, double z, int scale) {
    return dx(y, x, z, scale);
  }

  double dz(double x, double y, double z, int scale) {
    return dx(z, y, x, scale);
  }

  // 3D second order gaussian derivatives
  double dxx(double x, double y, double z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_seventh = scale_squared * scale_squared * scale_squared * scale;
    return 
      (x*x - scale_squared) * 
      (const_normalise / scale_seventh) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  double dyy(double x, double y, double z, int scale) {
    return dxx(y, x, z, scale);
  }

  double dzz(double x, double y, double z, int scale) {
    return dxx(z, y, x, scale);
  }

  double dxy(double x, double y, double z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_seventh = scale_squared * scale_squared * scale_squared * scale;
    return 
      x * y *
      (const_normalise / scale_seventh) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  double dxz(double x, double y, double z, int scale) {
    return dxy(x, z, y, scale);
  }

  double dyz(double x, double y, double z, int scale) {
    return dxy(y, z, x, scale);
  }


  //
  // Kernel functions
  //

  // 1D Kernel
  std::vector<double> 
  kernel_1d(double (&filter_function)(double, int), int scale) {
    std::vector<double> kernel;
    int width = 2 * 3 * scale + 1;
    kernel.reserve(width);

    for (double x = -3*scale; x <= 3*scale; x += 1) {
      kernel.push_back(filter_function(x, scale));
    }
    return kernel;
  }

  // 2D Kernel
  std::vector< std::vector<double> > 
  kernel_2d(double (&filter_function)(double, double, int), int scale) {
    int width = 2 * 3 * scale + 1;

    std::vector< std::vector<double> > kernel;
    std::vector<double> ys;
    kernel.reserve(width);
    ys.resize(width);

    for (double x = -3*scale; x <= 3*scale; x += 1) {
      int i = 0;
      for (double y = -3*scale; y <= 3*scale; y += 1) {
	ys[i++] = filter_function(x, y, scale);
      }
      kernel.push_back(ys);
    }

    return kernel;
  }

  
  // 3D Kernel
  std::vector< std::vector< std::vector<double> > > 
  kernel_3d(double (&filter_function)(double, double, double, int), int scale) {
    int width = 2 * 3 * scale + 1;

    std::vector< std::vector< std::vector<double> > > kernel;
    std::vector< std::vector<double> > ys;
    std::vector<double> zs;
    kernel.reserve(width);
    ys.resize(width);
    zs.resize(width);

    for (double x = -3*scale; x <= 3*scale; x += 1) {
      int i = 0;
      for (double y = -3*scale; y <= 3*scale; y += 1) {
	int j = 0;
	for (double z = -3*scale; z <= 3*scale; z += 1) {     
	  zs[j++] = filter_function(x, y, z, scale);
	}
	ys[i++] = zs;
      }
      kernel.push_back(ys);
    }

    return kernel;
  }

}
