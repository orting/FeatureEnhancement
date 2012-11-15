#pragma once
#include <functional>
#include "Volume.h"

namespace feature_enhancement {
  //  typedef std::function<double (double, int)> Filter1D;
  //  typedef std::function<double (double, double, int)> Filter2D;
  typedef std::function<double (double, double, double, int)> Filter3D;

  void kernel(Filter3D filter, int scale, Volume &out);
}
