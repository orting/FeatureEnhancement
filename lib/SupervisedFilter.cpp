#include <utility>
#include <array>
#include <complex>
#include <cstdio>
#include <fftw3.h>
#include <math.h>

#include "pechin_wrap.h"

#include "GaussFilter.h"
#include "SupervisedFilter.h"
#include "Util.h"

using namespace feature_enhancement;

SupervisedFilter::SupervisedFilter():
  trained(false), feature_matrix(), stored(), classifier()
{}


//Apply the filter to the given volume
void SupervisedFilter::apply(cimg_library::CImg<short> &volume, 
			     cimg_library::CImg<unsigned char> const &segmentation) {
  
  if (! this->trained) {
    std::cerr << "Filter has not been trained. Aborting" << std::endl;
    return;
  }
  this->calculate_features(volume);
  this->classify(volume);
}


// Classify the voxels
void SupervisedFilter::classify(cimg_library::CImg<short> &classified) {
  int k = 15;
  cimg_forXYZ(classified, x, y, z) {
    std::vector<short> features = this->get_feature_vector(x, y, z);
    std::vector<int> nn = this->classifier.knn(features, k);
    int classification = 0;
    for (auto n : nn) {
      classification += n;
    }
    classified(x,y,z) = (classification * 1000) / 15; // Find something better
  }
}


// train the filter
void SupervisedFilter::train(std::vector<std::string> const &filenames) {
  std::vector< std::vector<short> > classification_matrix;

  int point_number = 0;
  for (auto filename : filenames) {
    std::pair< std::string, std::vector<Point4D> > parsed = parse(filename);
    cimg_library::CImg<short> volume(parsed.first.c_str());
    this->calculate_features(volume);

    classification_matrix.resize(classification_matrix.size() + parsed.second.size());

    for (int scale = 0; scale < 4; ++scale) {
      for (Feature f = Feature::Gauss; f < Feature::OutOfBounds; ++f) {
	if (this->feature_matrix[f][scale]) {
	  int j = point_number;
	  for (auto point : parsed.second) {
	    short feature = this->get_feature(f, scale, point.x, point.y, point.z);
	    classification_matrix[j++].push_back(feature);
	  }
	}
      }
    }
    for (auto point : parsed.second) {
      classification_matrix[point_number++].push_back(point.c);
    }
  }

  this->classifier.set_classification_matrix(classification_matrix);
  this->classifier.save("something.classification");
  this->trained = true;
}


void SupervisedFilter::add_feature(Feature feature, int scale) {
  if ( scale >= 0 && scale < 4) {
    this->feature_matrix[feature][scale] = true;
  }
}
void SupervisedFilter::remove_feature(Feature feature, int scale) {
  if (scale >= 0 && scale < 4) {
    this->feature_matrix[feature][scale] = false;
  }
}


// Store the calculated feature in a open temporary file, the filehandle is stored in
// this->stored. Should abstract the storage so disk is only used when RAM is insufficient
void SupervisedFilter::store(Feature f, int scale, cimg_library::CImg<short> const &vol) {
  FILE * file = std::tmpfile(); // files created with tmpfile are cleaned up automagicallly

  // This is probably not necesary
  if (this->stored[f][scale].second != 0) {
    std::fclose(this->stored[f][scale].second); 
  }
  this->stored[f][scale].first.x = vol.width();
  this->stored[f][scale].first.y = vol.height();
  this->stored[f][scale].first.z = vol.depth();
  this->stored[f][scale].second = file;
  std::fwrite((const void *)vol.data(), sizeof(short), vol.width() * vol.height() * vol.depth(), file);
}


bool SupervisedFilter::is_stored(Feature f, int scale) {
  return this->stored[f][scale].second != 0;
}



// Calculates the necesary features
void SupervisedFilter::calculate_features(cimg_library::CImg<short> &volume, 
					  cimg_library::CImg<unsigned char> const &segmentation) {
  BoundingCube cube = get_bounding_cube(segmentation);
  cimg_library::CImg<short> cropped(volume.get_crop(cube.start_x, cube.start_y, cube.start_z,
						    cube.end_x, cube.end_y, cube.end_z));
  calculate_features(cropped);
}

void SupervisedFilter::calculate_features(cimg_library::CImg<short> &volume) {
  if (this->feature_matrix[Feature::Identity][0]) {
    this->store(Feature::Identity, 0, volume);
  }

  GaussFilter3D gauss;
  for (int i = 0; i < 4; ++i) {
    int scale = std::pow(2, i); // Temporary hack, not clear what is actually needed
    int j;
    // Gradient
    if (this->feature_matrix[Feature::Gradient][i]) {
      cimg_library::CImgList<short> first_order(3, volume);
      cimg_library::CImg<short> gradient(volume);
      gauss.apply_dx(first_order(0), scale);
      gauss.apply_dy(first_order(1), scale);
      gauss.apply_dz(first_order(2), scale);
      cimg_forXYZ(gradient, x, y, z) {
	gradient(x,y,z) = calculate_gradient(first_order(0)(x,y,z), 
					     first_order(1)(x,y,z), 
					     first_order(2)(x,y,z));
      }
      this->store(Feature::Gradient, scale, gradient);
      j = 0;
      for (Feature f = Feature::GaussDx; f <= Feature::GaussDz; ++f, ++j) {
	if (this->feature_matrix[f][i]) {
	  this->store(f, i, first_order(j));
	}
      }
    }

    // Eigenvalues of the hessian
    if (this->feature_matrix[Feature::HessianEig1][i] ||
	this->feature_matrix[Feature::HessianEig2][i] || 
	this->feature_matrix[Feature::HessianEig3][i] ) {
      cimg_library::CImgList<short> second_order(6, volume);
      gauss.apply_dxx(second_order(0), scale);
      gauss.apply_dxy(second_order(1), scale);
      gauss.apply_dxz(second_order(2), scale);
      gauss.apply_dyy(second_order(3), scale);
      gauss.apply_dyz(second_order(4), scale);
      gauss.apply_dzz(second_order(5), scale);

      std::array<double, 6> hessian;
      std::array<double, 3> eigenvalues;
      cimg_library::CImgList<short> eigens;
      j = 0;
      for (Feature f = Feature::HessianEig1; f <= Feature::HessianEig3; ++f, ++j) {
	if (this->feature_matrix[f][i]) {
	  eigens(j).resize(volume);
	}
      }
      cimg_forXYZ(second_order(0), x, y, z) {
	for (j = 0; j < 6; ++j) {
	  hessian[j] = second_order(j)(x,y,z);
	}
	calculate_eigenvalues(hessian, eigenvalues);

	j = 0;
	for (Feature f = Feature::HessianEig1; f <= Feature::HessianEig3; ++f, ++j) {
	  if (this->feature_matrix[f][i]) {
	    eigens(j)(x,y,z) = eigenvalues[j];
	  }
	}
      }
      
      j  = 0;
      for (Feature f = Feature::HessianEig1; f <= Feature::HessianEig3; ++f, ++j) {
	if (this->feature_matrix[f][i]) {
	  this->store(f, i, eigens(j));
	}
      }

      j = 0;
      for (Feature f = Feature::GaussDxx; f <= Feature::GaussDzz; ++f, ++j) {
	if (this->feature_matrix[f][i]) {
	  this->store(f, i, second_order(j));
	}
      }
    }
    // I really need to find a nice way of doing this
    std::vector<GaussFilter3D::FilterFunction> gfs;
    gfs.resize(Feature::GaussDzz + 1);
    gfs[Feature::Gauss] = &GaussFilter3D::gauss;
    gfs[Feature::GaussDx] = &GaussFilter3D::dx;
    gfs[Feature::GaussDy] = &GaussFilter3D::dy;
    gfs[Feature::GaussDz] = &GaussFilter3D::dz;
    gfs[Feature::GaussDxx] = &GaussFilter3D::dxx;
    gfs[Feature::GaussDxy] = &GaussFilter3D::dxy;
    gfs[Feature::GaussDxz] = &GaussFilter3D::dxz;
    gfs[Feature::GaussDyy] = &GaussFilter3D::dyy;
    gfs[Feature::GaussDyz] = &GaussFilter3D::dyz;
    gfs[Feature::GaussDzz] = &GaussFilter3D::dzz;

    for (Feature f = Feature::Gauss; f <= Feature::GaussDzz; ++f) {
      if (this->feature_matrix[f][i] && !this->is_stored(f, i)) {
	cimg_library::CImg<short> current(volume);
	gauss.apply(current, scale, gfs[f]);
	this->store(f, i, current);
      }
    }
  }
}


// Assumes a file with this format:
// path/to/volume
// x1 y1 z1 c1
// ...
// xn yn zn cn 
std::pair< std::string, std::vector<Point4D> > SupervisedFilter::parse(std::string filename) {
  std::pair< std::string, std::vector<Point4D> > result;

  std::ifstream file;
  file.open(filename, std::ifstream::in);

  file >> result.first; // name of volume

  Point4D point;
  while(file.good()) {
    file >> point.x
	 >> point.y
	 >> point.z
	 >> point.c;
    result.second.push_back(point);
  }

  return result;
}



// column-major order.
long to_offset(int x, int y, int z, int w, int h) {
  return x + y * w + z * w * h;
}


short SupervisedFilter::get_feature(Feature f, int scale, int x, int y, int z) {
  // assert(x < this->stored[f][scale].x && y < this->stored[f][scale].y && z this->stored[f][scale].z);
  long offset = to_offset(x, y, z, this->stored[f][scale].first.x, this->stored[f][scale].first.y);
  fseek(this->stored[f][scale].second, sizeof(short) * offset, SEEK_SET);

  short feature;
  fread(&feature, sizeof(short), 1, this->stored[f][scale].second);
  return feature;
}


std::vector<short> SupervisedFilter::get_feature_vector(int x, int y, int z) {
  std::vector<short> features;
  if (this->feature_matrix[Feature::Identity][0]) {
    features.push_back(get_feature(Feature::Identity, 0, x, y, z));
  }
  for (int i = 0; i < 4; ++i) {
    for (Feature f = Feature::Gauss; f < Feature::OutOfBounds; ++f) {
      features.push_back(get_feature(f, i, x, y, z));
    }
  }

  return features;
}
