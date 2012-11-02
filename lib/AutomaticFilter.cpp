#include <math.h>
#include <array>

#include "pechin_wrap.h"

#include "AutomaticFilter.h"
#include "FeatureMeasure.h"
#include "Filter.h"
#include "Gauss.h"
#include "Util.h"

using namespace feature_enhancement;


typedef std::vector< std::vector< std::vector<double> > > Vector3D;

AutomaticFilter::AutomaticFilter():
  feature_meassure([&](double voxel, double a, double b, double c) {c=a=b=c; return voxel;})
{}


void AutomaticFilter::apply(cimg_library::CImg<short> &volume, 
			    cimg_library::CImg<unsigned char> const &segmentation,
			    double threshhold,
			    int scale,
			    bool use_fft) {
  if (use_fft) {
    this->apply_fft(volume, segmentation, threshhold, scale);
  }
  else {
    this->apply_no_fft(volume, segmentation, threshhold, scale);
  }
}

void AutomaticFilter::apply(cimg_library::CImg<short> &volume, 
			    double threshhold,
			    int scale,
			    bool use_fft) {
  cimg_library::CImg<unsigned char> segmentation(volume, "xyz", 1);
  this->apply(volume, segmentation, threshhold, scale, use_fft);
}



void AutomaticFilter::set_featureness(FeatureMeassure feature_meassure) {
  this->feature_meassure = feature_meassure;
}

void AutomaticFilter::apply_fft(cimg_library::CImg<short> &volume, 
				cimg_library::CImg<unsigned char> const &segmentation,
				double threshhold,
				int scale) {
  cimg_library::CImgList<double> features(6, volume);
  filter::apply(features(0), scale, gauss::dxx);
  filter::apply(features(1), scale, gauss::dxy);
  filter::apply(features(2), scale, gauss::dxz);
  filter::apply(features(3), scale, gauss::dyy);
  filter::apply(features(4), scale, gauss::dyz);
  filter::apply(features(5), scale, gauss::dzz);

  std::array<double, 6> hessian;
  std::array<double, 3> eigenvalues;
  double featureness;
  cimg_forXYZ(features(0), x, y, z) {
    if (segmentation(x, y, z) == 0) {
      volume(x, y, z) = -1000;
    }
    else {
      hessian[0] = features(0)(x,y,z);
      hessian[1] = features(1)(x,y,z);
      hessian[2] = features(2)(x,y,z);
      hessian[3] = features(3)(x,y,z);
      hessian[4] = features(4)(x,y,z);
      hessian[5] = features(5)(x,y,z);
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


void AutomaticFilter::apply_no_fft(cimg_library::CImg<short> &volume, 
				   cimg_library::CImg<unsigned char> const &segmentation,
				   double threshhold,
				   int scale) {
  cimg_library::CImg<short> feature(volume, "xyz", -1000);
  
  Vector3D dxx(filter::kernel_3d(gauss::dxx, scale));
  Vector3D dxy(filter::kernel_3d(gauss::dxy, scale));
  Vector3D dxz(filter::kernel_3d(gauss::dxz, scale));
  Vector3D dyy(filter::kernel_3d(gauss::dyy, scale));
  Vector3D dyz(filter::kernel_3d(gauss::dyz, scale));
  Vector3D dzz(filter::kernel_3d(gauss::dzz, scale));
 
  std::array<double,6> hessian;
  std::array<double, 3> eigenvalues;
  double featureness;
  short voxel_value;

  int width = 3 * scale; // ± 3 standard deviations
  cimg_for_insideXYZ(volume, x, y, z, width) {
    if (segmentation(x,y,z) > 0) {
      voxel_value = volume(x,y,z);
      hessian[0] = hessian[1] = hessian[2] = hessian[3] = hessian[4] = hessian[5] = 0;
      for (int ii = 0, i = x - width; i <= x + width; ++i, ++ii) {
	for (int jj = 0, j = y - width; j <= y + width; ++j, ++jj) {
	  for (int kk = 0, k = z - width; k <= z + width; ++k, ++kk) {
	    hessian[0] +=  volume(i, j, k) * dxx[ii][jj][kk];
	    hessian[1] +=  volume(i, j, k) * dxy[ii][jj][kk];
	    hessian[2] +=  volume(i, j, k) * dxz[ii][jj][kk];
	    hessian[3] +=  volume(i, j, k) * dyy[ii][jj][kk];
	    hessian[4] +=  volume(i, j, k) * dyz[ii][jj][kk];
	    hessian[5] +=  volume(i, j, k) * dzz[ii][jj][kk];
	  }
	}
      }
      calculate_eigenvalues(hessian, eigenvalues);
      if (eigenvalues[2] < 0) {
	featureness = (this->feature_meassure)(voxel_value, eigenvalues[0], eigenvalues[1], eigenvalues[2]);
	if (featureness > threshhold) {
	  feature(x,y,z) = voxel_value;
	}
      }
    }
  }
  volume.assign(feature);
}
