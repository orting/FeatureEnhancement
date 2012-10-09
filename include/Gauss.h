#pragma once
#include <vector>
#include <math.h>

namespace gauss {
  // 1D
  double gauss(double x, int scale);
  double dx(double x, int scale);

  // 2D
  double gauss(double x, double y, int scale);
  double dx(double x, double y, int scale);
  double dy(double x, double y, int scale);
  double dxx(double x, double y, int scale);
  double dyy(double x, double y, int scale);
  double dxy(double x, double y, int scale);

  // 3D
  double gauss(double x, double y, double z, int scale);
  double dx(double x, double y, double z, int scale);
  double dy(double x, double y, double z, int scale);
  double dz(double x, double y, double z, int scale);
  double dxx(double x, double y, double z, int scale);
  double dyy(double x, double y, double z, int scale);
  double dzz(double x, double y, double z, int scale);
  double dxy(double x, double y, double z, int scale);
  double dxz(double x, double y, double z, int scale);
  double dyz(double x, double y, double z, int scale);

  // Gaussian Kernels. They are symmetric in size
  // 1D 
  std::vector<double> 
  kernel_1d(double (&filter_function)(double, int), int scale);

  // 2D 
  std::vector< std::vector<double> >
  kernel_2d(double (&filter_function)(double, double, int), int scale);
  
  // 3D 
  std::vector< std::vector< std::vector<double> > > 
  kernel_3d(double (&filter_function)(double, double, double, int), int scale);
}
