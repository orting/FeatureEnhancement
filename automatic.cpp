#include <iostream>
#include <functional>
#include "pechin_wrap.h"
#include "AutomaticFilter.h"

int main(int argc, char *argv[]) {
  cimg_usage("Apply a fissure enhancement filter to the volume");
  const char* infile      = cimg_option("-i", (char*)0, "Input volume file. Expects 16-bit int");
  const char* outfile     = cimg_option("-o", (char*)0, "Output volume file");
  //  const char* seg_infile  = cimg_option("-is", (char*)0, "Input volume file with region of interest segmented,"
  //					"expects unsigned char values");
  const double threshhold = cimg_option("-t", 0.3, "Threshhold for featureness");
  const int scale         = cimg_option("-s", 1, "Scale for gauss derivatives");
  const int iterations    = cimg_option("-it", 1, "Number of times the algorithm should de run");
  const int featureness   = cimg_option("-f", 1, "1 = Rikxoort, 2 = Lassen, 3 = (Rikxoort + Lassen)/2");

  const double alpha = cimg_option("-alpha", 50, "Parameter for Lassens meassures");
  const double beta  = cimg_option("-beta", 35, "Parameter for Lassens meassures");
  const double gamma = cimg_option("-gamma", 25, "Parameter for Lassens meassures");
  
  const double hounsfield_mean = cimg_option("-hounsfield_mean", -484, "Parameter for Rikxoorts meassures");
  const double hounsfield_sd   = cimg_option("-hounsfield_sd", 407, "Parameter for Rikxoorts meassures");

  const size_t threads = cimg_option("-threads", 1, "Number of threads to use");
  
  if (infile == 0) {
    std::cerr << "Missing input volume. Use -h to get help" << std::endl;
    return -1;
  }

  using namespace feature_enhancement;
  
  AutomaticFilter filter(threads);
  switch (featureness) {
  case 2: {
    using namespace std::placeholders;
    auto meassure = std::bind(fissureness_lassen, alpha, beta, gamma, _2, _3, _4);
    filter.set_feature_measure(meassure);
    break;
  }
  case 3: {
    auto meassure = [&](double voxel, double eig1, double eig2, double eig3) {
      auto lassen = fissureness_lassen(alpha, beta, gamma, eig1, eig2, eig3);
      auto rikxoort = fissureness_rikxoort(hounsfield_mean, hounsfield_sd, voxel, eig1, eig2, eig3);
      return (lassen + rikxoort) / 2;
    };
    filter.set_feature_measure(meassure);
    break;
  }
  default: {
    using namespace std::placeholders;
    auto meassure = std::bind(fissureness_rikxoort, hounsfield_mean, hounsfield_sd, _1, _2, _3, _4);
    filter.set_feature_measure(meassure);
    break;
  }
  }

  cimg_library::CImg<short> cimg_volume(infile);
  Volume volume(cimg_volume.width(), cimg_volume.height(), cimg_volume.depth());
  std::cout << "Copying to Volume\n";
  cimg_forXYZ(cimg_volume, x, y, z) {
    volume(x, y, z) = cimg_volume(x, y, z);
  }
  // if (seg_infile != 0) {
  //   cimg_library::CImg<unsigned char> segmentation(seg_infile);

  //   for (int i = 0; i < iterations; ++i) {
  //     filter.apply(volume, segmentation, threshhold, scale, use_fft);
  //   }
  // }
  // else {
	      std::cout << "Applying filter\n";
    for (int i = 0; i < iterations; ++i) {
      filter.apply(volume, threshhold, scale);
    }
  // }

  std::cout << "Copying from Volume\n";
  cimg_forXYZ(cimg_volume, x, y, z) {
    cimg_volume(x, y, z) = volume(x, y, z);
  }

  if (outfile) {
    std::cout << "saving\n";
    cimg_volume.save(outfile);
  }
  else {
    std::cout << "displaying\n!";
    cimg_volume.display();
  }

  return 0;
}
