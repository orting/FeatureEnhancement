#include <iostream>
#include "pechin_wrap.h"
#include "AutomaticFilter.h"

int main(int argc, char *argv[]) {
  cimg_usage("Apply a feature enhancement filter to the volume");
  const char* infile      = cimg_option("-i", (char*)0, "Input volume file. Expects 16-bit int");
  const char* outfile     = cimg_option("-o", (char*)0, "Output volume file");
  const char* seg_infile  = cimg_option("-is", (char*)0, "Input volume file with region of interest segmented,"
					"expects unsigned char values");
  const double threshhold = cimg_option("-t", 0.3, "Threshhold for featureness");
  const int scale         = cimg_option("-s", 1, "Scale for gauss derivatives");
  const bool use_fft      = cimg_option("-fft", false, "Set whether to use FFT");
  const int iterations    = cimg_option("-it", 1, "Number of times the algorithm should de run");
  const int featureness   = cimg_option("-f", 1, "1 = Rikxoort, 2 = Lassen, 3 = (Rikxoort + Lassen)/2");

  if (infile == 0) {
    std::cerr << "Missing input volume. Use -h to get help" << std::endl;
    return -1;
  }

  feature_enhancement::AutomaticFilter filter;
  switch (featureness) {
  case 2:
    filter.set_featureness(&feature_enhancement::AutomaticFilter::fissureness_lassen);
    break;
  case 3:
    filter.set_featureness(&feature_enhancement::AutomaticFilter::fissureness_rikxoort_lassen);
    break;
  default:
    filter.set_featureness(&feature_enhancement::AutomaticFilter::fissureness_rikxoort);
    break;
  }

  cimg_library::CImg<short> volume(infile);
  if (seg_infile != 0) {
    cimg_library::CImg<unsigned char> segmentation(seg_infile);

    for (int i = 0; i < iterations; ++i) {
      filter.apply(volume, segmentation, threshhold, scale, use_fft);
    }
  }
  else {
    for (int i = 0; i < iterations; ++i) {
      filter.apply(volume, threshhold, scale, use_fft);
    }
  }

  if (outfile) {
    std::cout << "saving\n";
    volume.save(outfile);
  }

  std::cout << "displaying\n!";
  volume.display();  

  return 0;
}
