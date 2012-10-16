#include <math.h>
#include <array>

#include "pechin_wrap.h"

#include "AutomaticFilter.h"
#include "Filter.h"
#include "Gauss.h"
#include "Util.h"

using namespace feature_enhancement;


typedef std::vector< std::vector< std::vector<double> > > Vector3D;

AutomaticFilter::AutomaticFilter():
  feature_meassure(&AutomaticFilter::fissureness_rikxoort)
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


double AutomaticFilter::fissureness_rikxoort_lassen(double voxel_value, double eig1, double eig2, double eig3) const {
  double rikxoort = fissureness_rikxoort(voxel_value, eig1, eig2, eig3);
  double lassen = fissureness_lassen(voxel_value, eig1, eig2, eig3);
  return (rikxoort + lassen) / 2;
}

/* Calculates a fissureness measure using the method described in
 * Rikxoort et al: Supervised Enhancement Filters: Application to Fissure Detection in Chest CT Scans
 */
double AutomaticFilter::fissureness_rikxoort(double voxel_value, double eig1, double eig2, double eig3) const {
  double const hounsfield_mean = -700;// https://en.wikipedia.org/wiki/Hounsfield_scale#The_HU_of_common_substances
  double const hounsfield_sd = 50; // From casual inspection of data
  
  // we need the numericaly largest of the other 2
  double abs_eig1 = fabs(eig1);
  double abs_eig2 = fabs(eig2);
  double abs_eig3 = fabs(eig3);
  if (abs_eig1 > abs_eig2) {
    abs_eig2 = abs_eig1;
  }

  double plateness = (abs_eig3 - abs_eig2) / (abs_eig3 + abs_eig2);

  double hounsfield_diff = fabs(voxel_value) - fabs(hounsfield_mean);
  double hounsfield = exp(- (hounsfield_diff * hounsfield_diff) / (2 * hounsfield_sd * hounsfield_sd));

  return plateness * hounsfield;
}

/* Calculates a fissure similarity meassure using the method described in
 * Lassen et al: AUTOMATIC SEGMENTATION OF LUNG LOBES IN CT IMAGES BASED ON FISSURES, VESSELS, AND BRONCHI
 * requires: eig1 >= eig2 >= eig3
 */
double AutomaticFilter::fissureness_lassen(double voxel_value, double eig1, double eig2, double eig3) const {
  voxel_value = voxel_value;
  // alpha, beta, gamma are empirical, see Lassen et al
  double const alpha = 50;
  double const beta = 35;
  double const gamma = 25;
  double const beta6 = beta * beta * beta * beta * beta * beta;
  double const gamma6 = gamma * gamma * gamma * gamma * gamma * gamma;

  // we need to use the numericaly largest of the other eigenvalues
  if (fabs(eig1) > fabs(eig2)) {
    eig2 = eig1;
  }

  //  double structure = exp(- pow(eig3 - alpha, 6) / beta6); From the paper, but makes no sense
  double structure = exp(- pow(eig3 + alpha, 6) / beta6);
  double sheet = exp(- pow(eig2, 6) / gamma6);

  double fissure = structure * sheet;

  return fissure;
}


void AutomaticFilter::apply_fft(cimg_library::CImg<short> &volume, 
				cimg_library::CImg<unsigned char> const &segmentation,
				double threshhold,
				int scale) {
  BoundingCube cube = get_bounding_cube(segmentation);
  cimg_library::CImgList<short> features(6, volume.get_crop(cube.start_x, cube.start_y, cube.start_z,
							    cube.end_x, cube.end_y, cube.end_z));
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
    int xx = cube.start_x + x;
    int yy = cube.start_y + y;
    int zz = cube.start_z + z;
    if (segmentation(xx, yy, zz) != 0) {
      hessian[0] = features(0)(x,y,z);
      hessian[1] = features(1)(x,y,z);
      hessian[2] = features(2)(x,y,z);
      hessian[3] = features(3)(x,y,z);
      hessian[4] = features(4)(x,y,z);
      hessian[5] = features(5)(x,y,z);
      calculate_eigenvalues(hessian, eigenvalues);
      if (eigenvalues[2] < 0) {
	featureness = (this->*feature_meassure)(volume(xx, yy, zz), eigenvalues[0], eigenvalues[1], eigenvalues[2]);
	if (featureness > threshhold) {
	  features(0)(x,y,z) = volume(xx, yy, zz);
	}
	else {
	  features(0)(x,y,z) = -1000;
	}
      }
      else {
	features(0)(x,y,z) = -1000;
      }
    }
    else {
      features(0)(x,y,z) = -1000;
    }
  }
  volume.fill(-1000);
  insert_at(features(0), volume, cube);
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
	featureness = (this->*feature_meassure)(voxel_value, eigenvalues[0], eigenvalues[1], eigenvalues[2]);
	if (featureness > threshhold) {
	  feature(x,y,z) = voxel_value;
	}
      }
    }
  }
  volume.assign(feature);
}
