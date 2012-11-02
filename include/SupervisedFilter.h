#pragma once
#include <array>
#include <cstdio>
#include <tuple>
#include <utility>
#include <flann/flann.hpp>

#include "pechin_wrap.h"

#include "Feature.h"

namespace feature_enhancement {
  size_t const MAX_SCALE = 4;

  class SupervisedFilter {
  public:
    SupervisedFilter();
    ~SupervisedFilter();


    //    void apply(cimg_library::CImg<short> &volume, size_t nn);
    void apply(cimg_library::CImg<short> &volume, std::string dataset_path, size_t nn);
    void apply(cimg_library::CImg<short> &volume, std::string dataset_path, std::string index_path, size_t nn);

    void add_feature(Feature feature, size_t scale);
    void remove_feature(Feature feature, size_t scale);

  private:
    void classify(cimg_library::CImg<short> &volume, flann::Index<flann::L2<short> > &index, size_t nn);

    void calculate_features(cimg_library::CImg<short> &volume);
    void store(Feature f, size_t scale, cimg_library::CImg<short> const &vol);
    void load_dataset(std::string dataset_path);
    flann::Index<flann::L2<short> > load_index(std::string index_path);
    flann::Index<flann::L2<short> > make_index(std::string save_path);

    size_t get_feature_index(Feature f, size_t scale);
    size_t used_features();
    void save_feature_matrices();
    void allocate_query(size_t voxels);

    std::array< std::array<bool, MAX_SCALE>, Feature::OutOfBounds> feature_matrix;
    std::array< std::array<bool, MAX_SCALE>, Feature::OutOfBounds> calculated_features;
    flann::Matrix<short> dataset;
    flann::Matrix<short> classifications;
    flann::Matrix<short> query;
  };
}
