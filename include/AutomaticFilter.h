#pragma once
#include "Transforms.h"
#include "Volume.h"
#include "FeatureMeasure.h"

namespace feature_enhancement {
  class AutomaticFilter {
  public:
    AutomaticFilter(size_t threads=1,
		    FeatureMeasure f=[&](double voxel, double a, double b, double c) {c=a=b=c; return voxel;});
    ~AutomaticFilter(){};

    void apply(Volume &volume,
	       double threshhold,
	       int scale = 1);

    // void apply(Volume &volume,
    // 	       Volume &segmentation,
    // 	       double threshhold,
    // 	       int scale = 1);

    void set_feature_measure(FeatureMeasure feature_meassure);

  private:
    FeatureMeasure feature_meassure;
    FFT fft;
  };

}
