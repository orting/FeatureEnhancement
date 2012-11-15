#include <array>

#include "Volume.h"
#include "VolumeList.h"
#include "Transforms.h"
#include "AutomaticFilter.h"
#include "FeatureMeasure.h"
#include "Filter.h"
#include "Gauss.h"
#include "Util.h"

namespace feature_enhancement {
  Filter3D dxx = dxx3D, dxy = dxy3D, dxz = dxz3D, dyy = dyy3D, dyz = dyz3D, dzz = dzz3D;

  AutomaticFilter::AutomaticFilter(size_t threads, FeatureMeasure f):
    feature_meassure(f),
    fft(threads)
  {}

  void AutomaticFilter::apply(Volume &volume, 
			      double threshhold,
			      int scale) {
    Volume copy(volume);
    fft.forward(copy);
    VolumeList filters(6, volume.width, volume.height, volume.depth);
    std::array<Filter3D, 6> ff = {{dxx, dxy, dxz, dyy, dyz, dzz}};

    for (size_t i = 0; i < filters.size(); ++i) {
      kernel(ff[i], scale, filters[i]);
    }

    fft.convolve(filters, copy);

    std::array<double, 6> hessian;
    std::array<double, 3> eigenvalues = {{0,0,0}};
    double featureness;
    for (size_t x = 0; x < volume.width; ++x) {
      for (size_t y = 0; y < volume.height; ++y) {
     	for (size_t z = 0; z < volume.depth; ++z) {
    	  for (size_t i = 0; i < 6; ++i) {
    	    hessian[i] = filters[i](x,y,z);
    	  }
     	  calculate_eigenvalues(hessian, eigenvalues);
    	  if (eigenvalues[2] >= 0) {
    	    volume(x, y, z) = -1000;
    	  } 
    	  else {
    	    featureness = (this->feature_meassure)(volume(x, y, z), eigenvalues[0], eigenvalues[1], eigenvalues[2]);
    	    if (featureness <= threshhold) {
    	      volume(x, y, z) = -1000;
    	    }
    	  }
     	}
      }
    }
  }

  void AutomaticFilter::set_feature_measure(FeatureMeasure f) {
    this->feature_meassure = f;
  }
}
