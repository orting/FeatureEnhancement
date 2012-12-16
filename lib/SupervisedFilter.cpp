#include <utility>
#include <array>
#include <complex>
#include <cstdio>
#include <fftw3.h>
#include <math.h>
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>


#include "Feature.h"
#include "Filter.h"
#include "Gauss.h"
#include "SupervisedFilter.h"
#include "Util.h"
#include "Transforms.h"
#include "Volume.h"
#include "VolumeList.h"

using namespace feature_enhancement;
Filter3D gauss = gauss3D, dx = dx3D, dy = dy3D, dz = dz3D, dxx = dxx3D, dxy = dxy3D, dxz = dxz3D, dyy = dyy3D, dyz = dyz3D, dzz = dzz3D;

SupervisedFilter::SupervisedFilter(size_t threads):
  feature_matrix(), calculated_features(), dataset(), classifications(), query(), fft(threads)
{}

SupervisedFilter::~SupervisedFilter() {
  delete[] this->query.ptr();
  delete[] this->dataset.ptr();
  delete[] this->classifications.ptr();  
}

void SupervisedFilter::apply(Volume &volume, std::string dataset_path, size_t nn) {
  this->calculate_features(volume);
  this->load_dataset(dataset_path);
  flann::Index<flann::L2<short> > index = this->make_index(dataset_path);
  this->classify(volume, index, nn);
}

void SupervisedFilter::apply(Volume &volume, std::string dataset_path, std::string index_path, size_t nn) {
  this->calculate_features(volume);
  this->load_dataset(dataset_path);
  flann::Index<flann::L2<short> > index = this->load_index(index_path);
  this->classify(volume, index, nn);
}


void SupervisedFilter::classify(Volume &volume, 
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
  for (size_t x = 0; x < volume.width; ++x) {
    for (size_t y = 0; y < volume.height; ++y) {
      for (size_t z = 0; z < volume.depth; ++z) {
	int feature_neighbours = 0;
	for (size_t j = 0; j < nn; ++j) {
	  feature_neighbours += this->classifications[indices[row][j]][0];
	}
	volume(x,y,z) = feature_neighbours; // What is a good meassure here?
	++row;
      }
    }
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


// Setup features for selection
// Brug et map
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


void SupervisedFilter::store(Feature f, size_t scale, Volume const &vol) {
  size_t i = get_feature_index(f, scale);
  size_t j = 0;
  for (size_t x = 0; x < vol.width; ++x) {
    for (size_t y = 0; y < vol.height; ++y) {
      for (size_t z = 0; z < vol.depth; ++z) {
	this->query[j++][i] = vol(x,y,z);
      }
    }
  }
}


void SupervisedFilter::calculate_features(Volume &volume) {
  this->allocate_query(volume.size);
  Volume copy(volume);

  if (this->feature_matrix[Feature::Identity][0]) {
    this->store(Feature::Identity, 0, volume);
  }

  for (size_t i = 0; i < MAX_SCALE; ++i) {
    size_t scale = std::pow(2, i); // Temporary hack, not clear what is actually needed
    size_t j = 0;
    // Gradient
    if (this->feature_matrix[Feature::Gradient][i]) {
      Volume gradient(volume.width, volume.height, volume.depth);
      VolumeList first_order(3, copy);
      kernel(dx, scale, first_order[0]);
      kernel(dy, scale, first_order[1]);
      kernel(dz, scale, first_order[2]);
      fft.convolve(first_order, copy);
      
      for (size_t x = 0; x < gradient.width; ++x) {
	for (size_t y = 0; y < gradient.height; ++y) {
	  for (size_t z = 0; z < gradient.depth; ++z) {
	    gradient(x,y,z) = calculate_gradient(first_order[0](x,y,z), first_order[1](x,y,z), first_order[2](x,y,z));
	  }
	}
      }
      this->store(Feature::Gradient, i, gradient);
      j = 0;
      for (Feature f = Feature::GaussDx; f <= Feature::GaussDz; ++f, ++j) {
	if (this->feature_matrix[f][i]) {
	  this->store(f, i, first_order[j]);
	}
      }
    }

    // Eigenvalues of the hessian
    if (this->feature_matrix[Feature::HessianEig1][i] ||
	this->feature_matrix[Feature::HessianEig2][i] || 
	this->feature_matrix[Feature::HessianEig3][i] ) {

      VolumeList second_order(6, copy);
      std::array<Filter3D, 6> second_order_functions = {{dxx, dxy, dxz, dyy, dyz, dzz}};
      for (size_t i = 0; i < second_order_functions.size(); ++i) {
	kernel(second_order_functions[i], scale, second_order[i]);
      }
      fft.convolve(second_order, copy);

      std::array<double, 6> hessian;
      std::array<double, 3> eigenvalues;
      VolumeList eigens(3, copy);

      for (size_t x = 0; x < copy.width; ++x) {
	for (size_t y = 0; y < copy.height; ++y) {
	  for (size_t z = 0; z < copy.depth; ++z) {
	    for (j = 0; j < 6; ++j) {
	      hessian[j] = second_order[j](x,y,z);
	    }
	    calculate_eigenvalues(hessian, eigenvalues);
	    j = 0;
	    for (Feature f = Feature::HessianEig1; f <= Feature::HessianEig3; ++f, ++j) {
	      if (this->feature_matrix[f][i]) {
		eigens[j](x,y,z) = eigenvalues[j];
	      }
	    }	    
	  }
	}
      }
  
      j  = 0;
      for (Feature f = Feature::HessianEig1; f <= Feature::HessianEig3; ++f, ++j) {
	if (this->feature_matrix[f][i]) {
	  this->store(f, i, eigens[j]);
	}
      }

      j = 0;
      for (Feature f = Feature::GaussDxx; f <= Feature::GaussDzz; ++f, ++j) {
	if (this->feature_matrix[f][i]) {
	  this->store(f, i, second_order[j]);
	}
      }
    }
    // I really need to find a nice way of doing this
    std::vector<Filter3D> ff;
    ff.resize(Feature::GaussDzz + 1);
    ff[Feature::Gauss] = gauss;
    ff[Feature::GaussDx] = dx;
    ff[Feature::GaussDy] = dy;
    ff[Feature::GaussDz] = dz;
    ff[Feature::GaussDxx] = dxx;
    ff[Feature::GaussDxy] = dxy;
    ff[Feature::GaussDxz] = dxz;
    ff[Feature::GaussDyy] = dyy;
    ff[Feature::GaussDyz] = dyz;
    ff[Feature::GaussDzz] = dzz;

    j = 0;
    for (Feature f = Feature::Gauss; f <= Feature::GaussDzz; ++f) {
      if (this->feature_matrix[f][i] && !this->calculated_features[f][i]) {
	++j;
      }
    }
    if (j > 0) {
      VolumeList missing(j, copy);
      j = 0;
      for (Feature f = Feature::Gauss; f <= Feature::GaussDzz; ++f) {
	if (this->feature_matrix[f][i] && !this->calculated_features[f][i]) {
	  kernel(ff[f], scale, missing[j++]);
	}
      }
      fft.convolve(missing, copy);
      j = 0;
      for (Feature f = Feature::Gauss; f <= Feature::GaussDzz; ++f) {
	if (this->feature_matrix[f][i] && !this->calculated_features[f][i]) {
	  this->store(f, i, missing[j++]);
	}
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
