#include "Feature.h"

namespace feature_enhancement {

  Feature& operator++(Feature& orig)  {
    if (orig < Feature::OutOfBounds) {
      orig = static_cast<Feature>(orig + 1);
    }
    else {
      orig = Feature::Identity;
    }
    return orig;
  }

  Feature operator++(Feature& orig, int) {
    Feature rVal = orig;
    ++orig;
    return rVal;
  }

}
