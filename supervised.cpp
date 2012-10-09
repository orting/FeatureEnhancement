#include <iostream>
#include "pechin_wrap.h"
#include "SupervisedFilter.h"

int main(int argc, char *argv[]) {
  cimg_usage("Apply a feature enhancement filter to the volume");
  const char* infile      = cimg_option("-i", (char*)0, "Input volume file. Expects 16-bit int values");
  const char* outfile     = cimg_option("-o", (char*)0, "Output volume file");
  const char* seg_infile  = cimg_option("-is", (char*)0, "Input volume file with region of interest segmented,"
					"expects unsigned char values");
  const char* training_file = cimg_option("-t", (char *)0, "Training file");
  //const char* featurefile     = cimg_option("-f", (char*)0, "File containing list of features to use");

  if (infile == 0 || seg_infile == 0) {
    std::cerr << "Not enough arguments. Use -h to get help" << std::endl;
    return -1;
  }

  std::vector<std::string> training_files;
  training_files.push_back(std::string(training_file));
  feature_enhancement::SupervisedFilter filter;
  filter.train(training_files);

  cimg_library::CImg<short> volume(infile);
  cimg_library::CImg<unsigned char> segmentation(seg_infile);

  filter.apply(volume, segmentation);
  std::cout << "displaying\n!";

  if (outfile) {
    volume.save(outfile);
  }
  volume.display();  

  return 0;
}
