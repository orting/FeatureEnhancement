#pragma once
#include "pechin_wrap.h"
#include "GaussFilter.h"

namespace feature_enhancement {
  class AutomaticFilter {
  public:
    typedef double (AutomaticFilter::*FeatureMeassure)(double, double, double, double) const;
    AutomaticFilter();
    ~AutomaticFilter();

    void apply(cimg_library::CImg<short> &volume,
	       cimg_library::CImg<unsigned char> const &segmentation,
	       double threshhold,
	       int scale = 1,
	       bool use_fft = false);

    void set_featureness(FeatureMeassure feature_meassure) {
      this->feature_meassure = feature_meassure;
    }

    double fissureness_rikxoort(double voxel_value, double eig1, double eig2, double eig3) const;
    double fissureness_lassen(double voxel_value, double eig1, double eig2, double eig3) const;
    double fissureness_rikxoort_lassen(double voxel_value, double eig1, double eig2, double eig3) const;

  private:
    void apply_no_fft(cimg_library::CImg<short> &volume,
		      cimg_library::CImg<unsigned char> const &segmentation,
		      double threshhold,
		      int scale);

    void apply_fft(cimg_library::CImg<short> &volume,
		   cimg_library::CImg<unsigned char> const &segmentation,
		   double threshhold,
		   int scale);
    
    FeatureMeassure feature_meassure;

  };

}
