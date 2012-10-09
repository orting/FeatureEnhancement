#pragma once
#include <array>
#include <cstdio>
#include <tuple>
#include <utility>
#include "Classifier.h"
#include "Filter.h"
#include "GaussFilter.h"
#include "pechin_wrap.h"

namespace feature_enhancement {
  enum Feature {
    Identity,
    Gauss,
    GaussDx,
    GaussDy,
    GaussDz,
    GaussDxx,
    GaussDxy,
    GaussDxz,
    GaussDyy,
    GaussDyz,
    GaussDzz,
    Gradient,
    HessianEig1,
    HessianEig2,
    HessianEig3,
    OutOfBounds
  };

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


  struct Point3D {
    int x, y, z;
  };

  struct Point4D {
    int x, y, z, c;
  };


  class SupervisedFilter {
  public:
    SupervisedFilter();
    ~SupervisedFilter(){};


    void apply(cimg_library::CImg<short> &volume);
    void apply(cimg_library::CImg<short> &volume,
    	       cimg_library::CImg<unsigned char> const &segmentation);

    void train(std::vector<std::string> const &training_files);

    void add_feature(Feature feature, int scale);
    void remove_feature(Feature feature, int scale);


  private:
    void classify(cimg_library::CImg<short> &classified_out);

    void calculate_features(cimg_library::CImg<short> &volume);
    void calculate_features(cimg_library::CImg<short> &volume,
			    cimg_library::CImg<unsigned char> const &segmentation);

    short get_feature(Feature f, int scale, int x, int y, int z);
    std::vector<short> get_feature_vector(int x, int y, int z);

    void store(Feature f, int scale, cimg_library::CImg<short> const &vol);
    bool is_stored(Feature f, int scale);

    std::pair< std::string, std::vector<Point4D> > parse(std::string filename);



    bool trained;
    std::array< std::array<bool, 4>, Feature::OutOfBounds> feature_matrix;
    std::array< std::array<std::pair<Point3D, FILE *>, 4>, Feature::OutOfBounds> stored;
    Classifier classifier;
  };

}



  /*
  enum Scale {
    A,
    B,
    C,
    D,
    OutOfBounds
  };

  Scale& operator++(Scale& orig)  {
    if (orig < Scale::OutOfBounds) {
      orig = static_cast<Scale>(orig + 1);
    }
    else {
      orig = Scale::A;
    }
    return orig;
  }

  Scale operator++(Scale& orig, int) {
    Scale rVal = orig;
    ++orig;
    return rVal;
  }*/


