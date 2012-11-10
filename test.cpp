#include <iostream>
#include <math.h>
#include <complex>
#include <fftw3.h>
#include "Volume.h"
#include "Transforms.h"
//#include "AutomaticFilter.h"
//#include "FeatureMeasure.h"


using namespace feature_enhancement;

bool test_volume();
bool test_volume_list();
bool test_volume_transform();
bool test_volume_list_transform();
bool test_automatic_filter();

int main() {
  std::cout << "Sizes:"
  	    << "\ndouble: " << sizeof(double)
  	    << "\nstd::complex<double>: " << sizeof(std::complex<double>)
  	    << "\nVolume: " << sizeof(Volume)
  	    << "\nVolumeList: " << sizeof(VolumeList)
    //  	    << "\nAutomaticFilter: " << sizeof(AutomaticFilter)
  	    << std::endl;


  std::cout << "Testing Volume ...\n";
  if (test_volume()) {
    std::cout << "... Volume test [OK]\n";
  } else {
    std::cout << "... Volume test [FAILED]\n";
  }

  std::cout << "Testing VolumeList ...\n";
  if (test_volume_list()) {
    std::cout << "... VolumeList test [OK]\n";
  } else {
    std::cout << "... VolumeList test [FAILED]\n";
  }


  std::cout << "Testing FFT on Volume ...\n";
  if (test_volume_transform()) {
    std::cout << "... FFT Volume test [OK]\n";
  } else {
    std::cout << "... FFT Volume test [FAILED]\n";
  }

  std::cout << "Testing FFT on VolumeList ...\n";
  if (test_volume_list_transform()) {
    std::cout << "... FFT VolumeList test [OK]\n";
  } else {
    std::cout << "... FFT VolumeList test [FAILED]\n";
  }

  //   std::cout << "Testing AutomaticFilter\n";
  // if (false // test_automatic_filter()
  //     ) {
  //   std::cout << "AutomaticFilter test [OK]\n";
  // } else {
  //   std::cout << "AutomaticFilter test [FAILED]\n";
  // }
    
  return 0;
}


bool test_volume() {
  const double epsilon = 0.0000000000001;
  const size_t w = 5;
  const size_t h = 10;
  const size_t d = 15;
  
  Volume vol(w, h, d);
  Volume inv(w, h, d);
  
  double i = 1;
  for (size_t x = 0; x < w; ++x) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t z = 0; z < d; ++z) {
	vol(x, y, z) = i++;
	inv(x, y, z) = 1 / vol(x, y, z);
      }
    }
  }
  
  inv *= vol;
  
  for (size_t x = 0; x < w; ++x) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t z = 0; z < d; ++z) {
	if (fabs(inv(x,y,z) - 1) > epsilon) {
	  std::cout << "error at index (" << x << ", " << y << ", " << z << ") "
		    << inv (x, y, z) << "\n";
	  return false;
	}
      }
    }
  }
  return true;
}

bool test_volume_list() {
  size_t n = 4, w = 6, h = 9, d = 7;
  double k = 1.0;
  VolumeList volumes(n, w, h, d);

  for (size_t x = 0; x < w; ++x) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t z = 0; z < d; ++z) {
	for (size_t i = 0; i < n; ++i) {
	  volumes[i](x, y, z) = k;
	}
	++k;
      }
    }
  }
  
  volumes[0] *= volumes[1];
  volumes[2] *= volumes[3];

  for (size_t x = 0; x < w; ++x) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t z = 0; z < d; ++z) {
	if (fabs(volumes[0](x,y,z) - volumes[2](x,y,z))) {
	  std::cout << "error at index (" << x << ", " << y << ", " << z << ") "
		    << volumes[0](x, y, z) << " : " << volumes[2](x, y, z) << "\n";
	  return false;
	}
      }
    }
  }

  return true;
  
}


bool test_volume_transform() {
  const double epsilon = 0.0000001;
  const size_t w = 150;
  const size_t h = 245;
  const size_t d = 367;
  
  Volume vol(w, h, d);

  double i = 0;
  for (size_t x = 0; x < w; ++x) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t z = 0; z < d; ++z) {
	vol(x, y, z) = i++;
      }
    }
  }

  FFT fft(4);
  fft.forward(vol);
  fft.backward(vol);

  i = 0;
  size_t errors = 0;
  double error = 0;
  double accumulated = 0;
  for (size_t x = 0; x < vol.width; ++x) {
    for (size_t y = 0; y < vol.height; ++y) {
      for (size_t z = 0; z < vol.depth; ++z) {
	error = fabs(i++ - vol(x, y, z));
  	if (error > epsilon) {
	  accumulated += error;
	  ++errors;
  	  // std::cout << "error at index (" << x << ", " << y << ", " << z << ") "
  	  // 	    << i << " : " << vol(x,y,z) << "\n";
  	  // return false;
  	}
      }
    }
  }
  if (errors > 0) {
    std::cout << errors << " errors totaling " << accumulated
	      << " giving an average of " << accumulated/errors << " from fft forward/backward\n";
    return false;
  }

  return true;
}


bool test_volume_list_transform() {
  const double epsilon = 0.0000000000001;
  size_t n = 3, w = 10, h = 20, d = 30;
  VolumeList volumes(n, w, h, d);

  double k = 0;
  for (size_t x = 0; x < w; ++x) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t z = 0; z < d; ++z) {
	for (size_t i = 0; i < n; ++i) {
	  volumes[i](x, y, z) = k;
	}
	++k;
      }
    }
  }

  // std::cout << "Before forward: \n";
  // for (size_t x = 0; x < w; ++x) {
  //   for (size_t y = 0; y < h; ++y) {
  //     for (size_t z = 0; z < d; ++z) {
  // 	std::cout << volumes.real_volumes[0](x, y, z) << " ";
  //     }
  //     std::cout << "\n";
  //   }
  //   std::cout << "\n";
  // }

  FFT fft(4);
  fft.forward(volumes);
  // std::cout << "After forward: \n";
  // for (size_t x = 0; x < w; ++x) {
  //   for (size_t y = 0; y < h; ++y) {
  //     for (size_t z = 0; z < d/2 +1; ++z) {
  // 	std::cout << volumes.complex_volumes[0](x, y, z) << " ";
  //     }
  //     std::cout << "\n";
  //   }
  //   std::cout << "\n";
  // }

  fft.backward(volumes);
  // std::cout << "After backward: \n";
  // for (size_t x = 0; x < w; ++x) {
  //   for (size_t y = 0; y < h; ++y) {
  //     for (size_t z = 0; z < d; ++z) {
  // 	std::cout << volumes.real_volumes[0](x, y, z) << " ";
  //     }
  //     std::cout << "\n";
  //   }
  //   std::cout << "\n";
  // }


  k = 0;
  for (size_t x = 0; x < w; ++x) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t z = 0; z < d; ++z, ++k) {
	for (size_t i = 0; i < n; ++i) {
	  if (fabs(volumes[i](x, y, z) - k) > epsilon) {
	    std::cout << "error at [" << i << "](" << x << ", " << y << ", " << z << ") "
		      << k << " : " << volumes[i](x, y, z) << "\n";
	    return false;
	  }
	}
      }
    }
  }
  return true;
}

// bool test_automatic_filter() {
//   size_t w = 512, h = 512, d = 367;
//   //size_t w = 10, h = 10, d = 10;
//    filter::Volume<double> vol(w, h, d);

//    for (size_t x = 0; x < w; ++x) {
//      for (size_t y = 0; y < h; ++y) {
//        for (size_t z = 0; z < d; ++z) {
// 	 vol(x, y, z) = static_cast<double>(x * y * z);
//        }
//      }
//    }

//    feature_enhancement::AutomaticFilter filter(4);
//   {
//     using namespace std::placeholders;
//     auto meassure = std::bind(feature_enhancement::fissureness_rikxoort, -500, 250, _1, _2, _3, _4);
//     filter.set_featureness(meassure);
//   }

//   filter.apply(vol, 0.1, 1);

//   return true;
// }
