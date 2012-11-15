#pragma once
#include <vector>
#include <math.h>

namespace feature_enhancement {
  // // 1D
  // double gauss(double x, int scale);
  // double dx(double x, int scale);

  // // 2D
  // double gauss(double x, double y, int scale);
  // double dx(double x, double y, int scale);
  // double dy(double x, double y, int scale);
  // double dxx(double x, double y, int scale);
  // double dyy(double x, double y, int scale);
  // double dxy(double x, double y, int scale);

  // // 3D
  double gauss3D(double x, double y, double z, int scale);
  double dx3D(double x, double y, double z, int scale);
  double dy3D(double x, double y, double z, int scale);
  double dz3D(double x, double y, double z, int scale);
  double dxx3D(double x, double y, double z, int scale);
  double dyy3D(double x, double y, double z, int scale);
  double dzz3D(double x, double y, double z, int scale);
  double dxy3D(double x, double y, double z, int scale);
  double dxz3D(double x, double y, double z, int scale);
  double dyz3D(double x, double y, double z, int scale);
}
