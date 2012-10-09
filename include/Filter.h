#pragma once
#include <vector>
#include "pechin_wrap.h"

namespace feature_enhancement {
  class Filter {};

  typedef void (Filter::*filter)(cimg_library::CImg<float>&, int, int, int, int);
  typedef void (Filter::*Filter2)(cimg_library::CImg<short>&, int);

  class HigherOrderFilter {
  public:
    cimg_library::CImgList<float> apply(cimg_library::CImgList<float>&);
    std::vector<filter> &get_requirements();
  };
}
