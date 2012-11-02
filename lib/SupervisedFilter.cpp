#include <utility>
#include <array>
#include <complex>
#include <cstdio>
#include <fftw3.h>
#include <math.h>
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include "pechin_wrap.h"

#include "Feature.h"
#include "Filter.h"
#include "Gauss.h"
#include "SupervisedFilter.h"
#include "Util.h"

using namespace feature_enhancement;

SupervisedFilter::SupervisedFilter():
  feature_matrix(), calculated_features(), dataset(), classifications(), query()
{}

SupervisedFilter::~SupervisedFilter() {
  delete[] this->query.ptr();
  delete[] this->dataset.ptr();
  delete[] this->classifications.ptr();  
}


// Loads data and classification from dataset, 
// builds a knn-index which is saved for future use.
//void SupervisedFilter::apply(cimg_library::CImg<short> &volume, size_t nn) {
//this->calculate_features(volume);
//this->classify(volume, nn);
//}

void SupervisedFilter::apply(cimg_library::CImg<short> &volume, std::string dataset_path, size_t nn) {
  this->calculate_features(volume);
  this->load_dataset(dataset_path);
  flann::Index<flann::L2<short> > index = this->make_index(dataset_path);
  this->classify(volume, index, nn);
}

void SupervisedFilter::apply(cimg_library::CImg<short> &volume, std::string dataset_path, std::string index_path, size_t nn) {
  this->calculate_features(volume);
  this->load_dataset(dataset_path);
  flann::Index<flann::L2<short> > index = this->load_index(index_path);
  this->classify(volume, index, nn);
}


void SupervisedFilter::classify(cimg_library::CImg<short> &volume, 
				flann::Index<flann::L2<short> > &index, 
				size_t nn) {
  // Need same number of points in dataset and classification
  // and same number of features in dataset and query.
  if (this->dataset.rows != this->classifications.rows || this->dataset.cols != this->query.cols) {
    std::cerr << "Error: Data dimensions mismatch" << std::endl;
    return;
  }

  flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
  flann::Matrix<float> distances(new float[query.rows*nn], query.rows, nn);

  std::cout << "Beginning nn search\n";
  index.knnSearch(query, indices, distances, nn, flann::SearchParams(128));

  std::cout << "Making probability image\n";
  size_t row = 0;
  cimg_forXYZ(volume, x, y, z) {
    short feature_neighbours = 0;
    for (size_t j = 0; j < nn; ++j) {
      feature_neighbours += this->classifications[indices[row][j]][0];
    }
    volume(x,y,z) = feature_neighbours; // What is a good meassure here?
    ++row;
  }

  delete[] indices.ptr();
  delete[] distances.ptr();
}


void SupervisedFilter::load_dataset(std::string dataset_path) {
  delete[] this->dataset.ptr();
  delete[] this->classifications.ptr();
  flann::load_from_file(this->dataset, dataset_path, "dataset");
  flann::load_from_file(this->classifications, dataset_path, "classifications");
}

flann::Index<flann::L2<short> > SupervisedFilter::load_index(std::string index_path) {
  return flann::Index<flann::L2<short> > (this->dataset, flann::SavedIndexParams(index_path));
}


flann::Index<flann::L2<short> > SupervisedFilter::make_index(std::string save_path) {
  flann::Index<flann::L2<short> > index (this->dataset, flann::AutotunedIndexParams(0.90, 0, 0, 1));
  std::cout << "building index\n";
  index.buildIndex();
  index.save(save_path + ".index");
  return index;
}


/*void SupervisedFilter::classify(cimg_library::CImg<short> &volume, std::string dataset_path, size_t nn) {
  delete[] this->dataset.ptr();
  delete[] this->classifications.ptr();
  flann::load_from_file(this->dataset, dataset_path, "dataset");
  flann::load_from_file(this->classifications, dataset_path, "classifications");
  
  flann::Index<flann::L2<short> > index(this->dataset, flann::AutotunedIndexParams(0.68, 0, 0, 1));
  std::cout << "building index\n";
  index.buildIndex();
  index.save(dataset_path + ".index");

  this->classify(volume, index, nn);
  }*/


// Setup features for selection
void SupervisedFilter::add_feature(Feature feature, size_t scale) {
  if (scale < MAX_SCALE) {
    this->feature_matrix[feature][scale] = true;
  }
}
void SupervisedFilter::remove_feature(Feature feature, size_t scale) {
  if (scale < MAX_SCALE) {
    this->feature_matrix[feature][scale] = false;
  }
}



size_t SupervisedFilter::get_feature_index(Feature f, size_t scale) {
  size_t i = 0;
  for (Feature f1 = Feature::Identity; f1 < Feature::OutOfBounds; ++f1) {
    for (size_t j = 0; j < MAX_SCALE; ++j) {
      if (f == f1 && scale == j) {
	return i; 
      }
      if (this->feature_matrix[f1][j]) {
	++i;
      }
    }
  }
  std::cerr << "Error: scale > MAX_SCALE" << std::endl; // Cant happen :)
  return i;
}

// Store the calculated feature in a open temporary file, the filehandle is stored in
// this->stored. Should abstract the storage so disk is only used when RAM is insufficient
void SupervisedFilter::store(Feature f, size_t scale, cimg_library::CImg<short> const &vol) {
  size_t i = get_feature_index(f, scale);
  size_t j = 0;
  cimg_forXYZ(vol, x, y, z) {
    this->query[j++][i] = vol(x,y,z);
  }
}



void SupervisedFilter::calculate_features(cimg_library::CImg<short> &volume) {
  this->allocate_query(volume.width() * volume.height() * volume.depth());

  if (this->feature_matrix[Feature::Identity][0]) {
    this->store(Feature::Identity, 0, volume);
  }

  for (size_t i = 0; i < MAX_SCALE; ++i) {
    size_t scale = std::pow(2, i); // Temporary hack, not clear what is actually needed
    int j;
    // Gradient
    if (this->feature_matrix[Feature::Gradient][i]) {
      cimg_library::CImgList<short> first_order(3, volume);
      cimg_library::CImg<short> gradient(volume);
      filter::apply(first_order(0), scale, gauss::dx);
      filter::apply(first_order(1), scale, gauss::dy);
      filter::apply(first_order(2), scale, gauss::dz);
      cimg_forXYZ(gradient, x, y, z) {
	gradient(x,y,z) = calculate_gradient(first_order(0)(x,y,z), 
					     first_order(1)(x,y,z), 
					     first_order(2)(x,y,z));
      }
      this->store(Feature::Gradient, i, gradient);
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
      filter::apply(second_order(0), scale, gauss::dxx);
      filter::apply(second_order(1), scale, gauss::dxy);
      filter::apply(second_order(2), scale, gauss::dxz);
      filter::apply(second_order(3), scale, gauss::dyy);
      filter::apply(second_order(4), scale, gauss::dyz);
      filter::apply(second_order(5), scale, gauss::dzz);

      std::array<double, 6> hessian;
      std::array<double, 3> eigenvalues;
      cimg_library::CImgList<short> eigens(3);
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
    std::vector<double(*)(double,double,double,int)> gfs;
    gfs.resize(Feature::GaussDzz + 1);
    gfs[Feature::Gauss] = &gauss::gauss;
    gfs[Feature::GaussDx] = &gauss::dx;
    gfs[Feature::GaussDy] = &gauss::dy;
    gfs[Feature::GaussDz] = &gauss::dz;
    gfs[Feature::GaussDxx] = &gauss::dxx;
    gfs[Feature::GaussDxy] = &gauss::dxy;
    gfs[Feature::GaussDxz] = &gauss::dxz;
    gfs[Feature::GaussDyy] = &gauss::dyy;
    gfs[Feature::GaussDyz] = &gauss::dyz;
    gfs[Feature::GaussDzz] = &gauss::dzz;

    for (Feature f = Feature::Gauss; f <= Feature::GaussDzz; ++f) {
      if (this->feature_matrix[f][i] && !this->calculated_features[f][i]) {
	cimg_library::CImg<short> current(volume);
	filter::apply(current, scale, *(gfs[f]));
	this->store(f, i, current);
      }
    }
  }
}


size_t SupervisedFilter::used_features() {
  size_t no_of_features = 0;
  for (Feature f = Feature::Identity; f < Feature::OutOfBounds; ++f) {
    for (size_t s = 0; s < MAX_SCALE; ++s) {
      if (this->feature_matrix[f][s]) {
	++no_of_features;
      }
    }
  }
  return no_of_features;
}


void SupervisedFilter::allocate_query(size_t voxels) {
  size_t features = used_features();
  delete[] query.ptr();
  this->query = flann::Matrix<short>(new short[voxels*features], voxels, features);
}
