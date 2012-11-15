#include <iostream>
#include <math.h>
#include <complex>
#include <fftw3.h>

#include "Gauss.h"
#include "Volume.h"
#include "VolumeList.h"
#include "Transforms.h"
#include "AutomaticFilter.h"
#include "FeatureMeasure.h"


using namespace feature_enhancement;

bool test_volume();
bool test_volume_list();
bool test_volume_transform();
bool test_volume_list_transform();
bool test_automatic_filter();
bool test_kernel();

int main() {
  std::cout << "Sizes:"
  	    << "\ndouble: " << sizeof(double)
  	    << "\nstd::complex<double>: " << sizeof(std::complex<double>)
  	    << "\nVolume: " << sizeof(Volume)
  	    << "\nVolumeList: " << sizeof(VolumeList)
      	    << "\nAutomaticFilter: " << sizeof(AutomaticFilter)
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

  std::cout << "Testing Kernel ...\n";
  if (test_kernel()) {
    std::cout << "... Kernel test [OK]\n";
  } else {
    std::cout << "... Kernel test [FAILED]\n";
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

  std::cout << "Testing AutomaticFilter\n";
  if (false //test_automatic_filter()
      ) {
    std::cout << "... AutomaticFilter test [OK]\n";
  } else {
    std::cout << "... AutomaticFilter test [FAILED]\n";
  }


    
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

  FFT fft(4);
  fft.forward(volumes);
  fft.backward(volumes);


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

bool test_automatic_filter() {
  size_t w = 512, h = 512, d = 367;
  //size_t w = 20, h = 20, d = 20;
  Volume vol(w, h, d);

   for (size_t x = 0; x < w; ++x) {
     for (size_t y = 0; y < h; ++y) {
       for (size_t z = 0; z < d; ++z) {
	 vol(x, y, z) = static_cast<double>(x * y * z);
       }
     }
   }

   using namespace std::placeholders;
   auto measure = std::bind(fissureness_rikxoort, -500, 250, _1, _2, _3, _4);
   AutomaticFilter filter(4, measure);

  filter.apply(vol, 0.1, 1);

  return true;
}


bool test_kernel() {
  double epsilon = 0.000001;
  const size_t w = 7, h = 7, d = 7;
  double dxx_kernel[w][h][d] = {
    {{6.96377e-07, 8.48361e-06, 3.80209e-05, 6.26859e-05, 3.80209e-05, 8.48361e-06, 6.96377e-07},
     {8.48361e-06, 0.000103352, 0.00046319, 0.000763671, 0.00046319, 0.000103352, 8.48361e-06},
     {3.80209e-05, 0.00046319, 0.00207587, 0.00342253, 0.00207587, 0.00046319, 3.80209e-05},
     {6.26859e-05, 0.000763671, 0.00342253, 0.0056428, 0.00342253, 0.000763671, 6.26859e-05},
     {3.80209e-05, 0.00046319, 0.00207587, 0.00342253, 0.00207587, 0.00046319, 3.80209e-05},
     {8.48361e-06, 0.000103352, 0.00046319, 0.000763671, 0.00046319, 0.000103352, 8.48361e-06},
     {6.96377e-07, 8.48361e-06, 3.80209e-05, 6.26859e-05, 3.80209e-05, 8.48361e-06, 6.96377e-07},
    },
    {{3.18136e-06, 3.87568e-05, 0.000173696, 0.000286376, 0.000173696, 3.87568e-05, 3.18136e-06},
     {3.87568e-05, 0.000472155, 0.00211605, 0.00348878, 0.00211605, 0.000472155, 3.87568e-05},
     {0.000173696, 0.00211605, 0.00948349, 0.0156356, 0.00948349, 0.00211605, 0.000173696},
     {0.000286376, 0.00348878, 0.0156356, 0.0257788, 0.0156356, 0.00348878, 0.000286376},
     {0.000173696, 0.00211605, 0.00948349, 0.0156356, 0.00948349, 0.00211605, 0.000173696},
     {3.87568e-05, 0.000472155, 0.00211605, 0.00348878, 0.00211605, 0.000472155, 3.87568e-05},
     {3.18136e-06, 3.87568e-05, 0.000173696, 0.000286376, 0.000173696, 3.87568e-05, 3.18136e-06},
    },
    {{0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
    },
    {{-7.83574e-06, -9.54588e-05, -0.000427817, -0.000705351, -0.000427817, -9.54588e-05, -7.83574e-06},
     {-9.54588e-05, -0.00116293, -0.00521188, -0.00859293, -0.00521188, -0.00116293, -9.54588e-05},
     {-0.000427817, -0.00521188, -0.023358, -0.0385108, -0.023358, -0.00521188, -0.000427817},
     {-0.000705351, -0.00859293, -0.0385108, -0.0634936, -0.0385108, -0.00859293, -0.000705351},
     {-0.000427817, -0.00521188, -0.023358, -0.0385108, -0.023358, -0.00521188, -0.000427817},
     {-9.54588e-05, -0.00116293, -0.00521188, -0.00859293, -0.00521188, -0.00116293, -9.54588e-05},
     {-7.83574e-06, -9.54588e-05, -0.000427817, -0.000705351, -0.000427817, -9.54588e-05, -7.83574e-06},
    },
    {{0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
    },
    {{3.18136e-06, 3.87568e-05, 0.000173696, 0.000286376, 0.000173696, 3.87568e-05, 3.18136e-06},
     {3.87568e-05, 0.000472155, 0.00211605, 0.00348878, 0.00211605, 0.000472155, 3.87568e-05},
     {0.000173696, 0.00211605, 0.00948349, 0.0156356, 0.00948349, 0.00211605, 0.000173696},
     {0.000286376, 0.00348878, 0.0156356, 0.0257788, 0.0156356, 0.00348878, 0.000286376},
     {0.000173696, 0.00211605, 0.00948349, 0.0156356, 0.00948349, 0.00211605, 0.000173696},
     {3.87568e-05, 0.000472155, 0.00211605, 0.00348878, 0.00211605, 0.000472155, 3.87568e-05},
     {3.18136e-06, 3.87568e-05, 0.000173696, 0.000286376, 0.000173696, 3.87568e-05, 3.18136e-06},
    },
    {{6.96377e-07, 8.48361e-06, 3.80209e-05, 6.26859e-05, 3.80209e-05, 8.48361e-06, 6.96377e-07},
     {8.48361e-06, 0.000103352, 0.00046319, 0.000763671, 0.00046319, 0.000103352, 8.48361e-06},
     {3.80209e-05, 0.00046319, 0.00207587, 0.00342253, 0.00207587, 0.00046319, 3.80209e-05},
     {6.26859e-05, 0.000763671, 0.00342253, 0.0056428, 0.00342253, 0.000763671, 6.26859e-05},
     {3.80209e-05, 0.00046319, 0.00207587, 0.00342253, 0.00207587, 0.00046319, 3.80209e-05},
     {8.48361e-06, 0.000103352, 0.00046319, 0.000763671, 0.00046319, 0.000103352, 8.48361e-06},
     {6.96377e-07, 8.48361e-06, 3.80209e-05, 6.26859e-05, 3.80209e-05, 8.48361e-06, 6.96377e-07}
    }};
  Volume vol(w,h,d);
  Filter3D dxx = dxx3D;
  kernel(dxx, 1, vol);

  for (size_t x = 0; x < w; ++x) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t z = 0; z < d; ++z) {
	if (fabs(vol(x, y, z) - dxx_kernel[x][y][z]) > epsilon) {
	  std::cout << "error at (" << x << ", " << y << ", " << z << ") "
		    <<  vol(x, y, z) << " : " << dxx_kernel[x][y][z] << "\n";
	  return false;
	}
      }
    }
  }
  return true;
}

