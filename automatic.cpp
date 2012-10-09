#include <iostream>
#include "pechin_wrap.h"
#include "AutomaticFilter.h"

int main(int argc, char *argv[]) {
  cimg_usage("Apply a feature enhancement filter to the volume");
  const char* infile      = cimg_option("-i", (char*)0, "Input volume file. Expects 16-bit int values");
  const char* outfile     = cimg_option("-o", (char*)0, "Output volume file");
  const char* seg_infile  = cimg_option("-is", (char*)0, "Input volume file with region of interest segmented,"
					"expects unsigned char values");
  const double threshhold = cimg_option("-t", 0.3, "Threshhold for featureness");
  const int scale         = cimg_option("-s", 1, "Scale for gauss derivatives");
  const bool use_fft      = cimg_option("-fft", false, "Set whether to use FFT");
  const int iterations    = cimg_option("-it", 1, "Number of times the algorithm should de run");
  //const int featureness   = cimg_option("-f", 1, "Use Rikxoort as measure");

  if (infile == 0 || seg_infile == 0) {
    std::cerr << "Not enough arguments. Use -h to get help" << std::endl;
    return -1;
  }

  feature_enhancement::AutomaticFilter filter;
  cimg_library::CImg<short> volume(infile);
  cimg_library::CImg<unsigned char> segmentation(seg_infile);

  for (int i = 0; i < iterations; ++i) {
    filter.apply(volume, segmentation, threshhold, scale, use_fft);
  }

  std::cout << "displaying\n!";

  if (outfile) {
    volume.save(outfile);
  }
  volume.display();  

  return 0;
}
