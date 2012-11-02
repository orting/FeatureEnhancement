#include <iostream>
#include <fstream>
#include "pechin_wrap.h"
#include "SupervisedFilter.h"

int main(int argc, char *argv[]) {
  cimg_usage("Apply a feature enhancement filter to the volume");
  const char* infile      = cimg_option("-i", (char*)0, "Input volume file. Expects 16-bit int values");
  const char* outfile     = cimg_option("-o", (char*)0, "Output volume file");
  const char* dataset     = cimg_option("-d", (char*)0, "Dataset in hdf5 format containing classified voxels."
					" Should contain 2 arrays. 1 named 'dataset' containing the data"
					" and 1 named 'classifications' containing the classifications.");
  const char* index       = cimg_option("-t", (char*)0, "Previously used index file");
  const size_t knn        = cimg_option("-k", 5, "Number of nearest neighbours to consider when classifying");
  
  //  const char* seg_infile  = cimg_option("-s", (char*)0, "Input volume file with region of interest segmented,"
  //					"expects unsigned char values");
  if (infile == 0) {
    std::cerr << "Not enough arguments. Use -h to get help" << std::endl;
    return -1;
  }
  cimg_library::CImg<short> volume(infile);
  if (dataset == 0) {
    cimg_forXYZ(volume, x, y, z) {
      if (volume(x,y,z) < 15) {
	volume(x,y,z) = 0;
      }
    }
    volume.display();
  }
  else {
    feature_enhancement::SupervisedFilter filter;

    filter.add_feature(feature_enhancement::Feature::Identity, 0);
    filter.add_feature(feature_enhancement::Feature::GaussDyz, 0);
    filter.add_feature(feature_enhancement::Feature::GaussDzz, 0);
    filter.add_feature(feature_enhancement::Feature::Gradient, 0);
    filter.add_feature(feature_enhancement::Feature::Gradient, 3);
    filter.add_feature(feature_enhancement::Feature::HessianEig1, 1);
    filter.add_feature(feature_enhancement::Feature::HessianEig1, 2);
    filter.add_feature(feature_enhancement::Feature::HessianEig2, 0);
    filter.add_feature(feature_enhancement::Feature::HessianEig2, 1);

    if (index != 0) {
      filter.apply(volume, dataset, index, knn);
    }
    else {
      filter.apply(volume, dataset, knn);
    }

    if (outfile) {
      volume.save(outfile);
    }
  }

  return 0;
}
