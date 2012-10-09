#pragma once
#include <vector>
#include <math.h>
#include <functional>

namespace feature_enhancement {
  //
  // Filter Functions
  //

  // 1D gaussian
  template<typename NumType>
  NumType 
  gauss(NumType x, int scale) {
    static double const const_normalise = 0.3989422804014327; // = 1 / sqrt(2*pi)
    return (const_normalise / scale) * exp((-(x*x)) / (2 * scale * scale));
  }

 // 1D first order gaussian derivatives
  template<typename NumType>
  NumType 
  gauss_dx(NumType x, int scale) {
    static double const const_normalise = 0.3989422804014327; // = 1 / sqrt(2*pi)
    double scale_squared = scale * scale;
    double scale_cubed = scale_squared * scale;
    return 
      -x * 
      (const_normalise / scale_cubed) * 
      exp((-(x*x)) / (2 * scale_squared));
  }


  // 2D gaussian
  template<typename NumType>
  NumType 
  gauss(NumType x, NumType y, int scale) {
    static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
    double scale_squared = scale * scale;
    return (const_normalise / scale_squared) * exp((-(x*x) -(y*y)) / (2 * scale_squared));
  }

  // 2D first order gaussian derivatives
  template<typename NumType>
  NumType 
  gauss_dx(NumType x, NumType y, int scale) {
    static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
    double scale_squared = scale * scale;
    double scale_fourth = scale_squared * scale_squared;
    return 
      -x * 
      (const_normalise / scale_fourth) * 
      exp((-(x*x) - (y*y)) / (2 * scale_squared));
  }

  template<typename NumType>
  NumType 
  gauss_dy(NumType x, NumType y, int scale) {
    return gauss_dx(y, x, scale);
  }


 // 2D second order gaussian derivatives
  template<typename NumType>
  NumType 
  gauss_dxx(NumType x, NumType y, int scale) {
    static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
    double scale_squared = scale * scale;
    double scale_sixth = scale_squared * scale_squared * scale_squared;
    return 
      (x*x - scale_squared) * 
      (const_normalise / scale_sixth) * 
      exp((-(x*x) - (y*y)) / (2 * scale_squared));
  }

  template<typename NumType>
  NumType 
  gauss_dyy(NumType x, NumType y, int scale) {
    return gauss_dxx(y, x, scale);
  }

  template<typename NumType>
  NumType 
  gauss_dxy(NumType x, NumType y, int scale) {
    static double const const_normalise = 0.15915494309189535; // = 1 / (2*pi)
    double scale_squared = scale * scale;
    double scale_sixth = scale_squared * scale_squared * scale_squared;
    return 
      x * y *
      (const_normalise / scale_sixth) * 
      exp((-(x*x) - (y*y)) / (2 * scale_squared));
  }


  // 3D gaussian
  template<typename NumType>
  NumType 
  gauss(NumType x, NumType y, NumType z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_cubed = scale_squared * scale;
    return (const_normalise / scale_cubed) * exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  // 3D first order gaussians derivatives
  template<typename NumType>
  NumType 
  gauss_dx(NumType x, NumType y, NumType z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_fifth = scale_squared * scale_squared * scale;
    return 
      -x * 
      (const_normalise / scale_fifth) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  template<typename NumType>
  NumType 
  gauss_dy(NumType x, NumType y, NumType z, int scale) {
    return gauss_dx(y, x, z, scale);
  }

  template<typename NumType>
  NumType 
  gauss_dz(NumType x, NumType y, NumType z, int scale) {
    return gauss_dx(z, y, x, scale);
  }


  // 3D second order gaussian derivatives
  template<typename NumType>
  NumType 
  gauss_dxx(NumType x, NumType y, NumType z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_seventh = scale_squared * scale_squared * scale_squared * scale;
    return 
      (x*x - scale_squared) * 
      (const_normalise / scale_seventh) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  template<typename NumType>
  NumType 
  gauss_dyy(NumType x, NumType y, NumType z, int scale) {
    return gauss_dxx(y, x, z, scale);
  }


  template<typename NumType>
  NumType 
  gauss_dzz(NumType x, NumType y, NumType z, int scale) {
    return gauss_dxx(z, y, x, scale);
  }


  template<typename NumType>
  NumType 
  gauss_dxy(NumType x, NumType y, NumType z, int scale) {
    static double const const_normalise = 0.06349363593424098; // = 1 / (sqrt(2*pi)^3)
    double scale_squared = scale * scale;
    double scale_seventh = scale_squared * scale_squared * scale_squared * scale;
    return 
      x * y *
      (const_normalise / scale_seventh) * 
      exp((-(x*x) - (y*y) - (z*z) ) / (2 * scale_squared));
  }

  template<typename NumType>
  NumType 
  gauss_dxz(NumType x, NumType y, NumType z, int scale) {
    return gauss_dxy(x, z, y, scale);
  }

  template<typename NumType>
  NumType 
  gauss_dyz(NumType x, NumType y, NumType z, int scale) {
    return gauss_dxy(y, z, x, scale);
  }


  //
  // Kernel functions
  //

  // 1D Kernel
  template<typename NumType>
  std::vector<NumType> 
  kernel(std::function<NumType(NumType, int)> filter_function, int scale) {
    std::vector<NumType> kernel;
    int width = 2 * 3 * scale + 1;
    kernel.reserve(width);

    for (NumType x = -3*scale; x <= 3*scale; x += 1) {
      kernel.push_back(filter_function(x, scale));
    }
    return kernel;
  }

  // 2D Kernel
  template<typename NumType>
  std::vector< std::vector<NumType> > 
  kernel(std::function<NumType(NumType, NumType, int)> filter_function, int scale) {
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
  template<typename NumType>
  std::vector< std::vector< std::vector<NumType> > > 
  kernel(std::function<NumType(NumType, NumType, NumType, int)> filter_function, int scale) {
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

}
