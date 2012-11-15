#include <unordered_map>
#include <iostream>
#include <complex>
#include <fftw3.h>
#include "Transforms.h"
#include "Volume.h"
#include "VolumeList.h"

namespace feature_enhancement {
  FFT::FFT(size_t threads)
    : forward_plans(),
      backward_plans()
  {
    if (!fftw_init_threads()) {
      std::cerr << "Unable to init threads in fftw\n";
      throw 1;
    }
    fftw_plan_with_nthreads(threads);
  }

  FFT::~FFT() {
    for (auto key_plan : forward_plans) {
      fftw_destroy_plan(key_plan.second);
    }
    for (auto key_plan : backward_plans) {
      fftw_destroy_plan(key_plan.second);
    }
    fftw_cleanup_threads();
  }

  void FFT::forward(Volume &vol) {
    if (vol.domain != Domain::Frequency) {
      fftw_plan plan;
      try {
	plan = forward_plans.at(vol.real_data);
      } catch (...) {
	plan = fftw_plan_dft_r2c_3d(vol.width, vol.height, vol.depth,
				    vol.real_data, 
				    reinterpret_cast<fftw_complex *>(vol.complex_data), 
				    FFTW_ESTIMATE);
	forward_plans[vol.real_data] = plan;
      }
      fftw_execute(plan);
      vol.domain = Domain::Frequency;
    }
  }

  void FFT::backward(Volume &vol) {   
    if (vol.domain != Domain::Time) {
      fftw_plan plan;
      try {
	plan = backward_plans.at(vol.real_data);
      } catch (...) {
	plan = fftw_plan_dft_c2r_3d(vol.width, vol.height, vol.depth,
				    reinterpret_cast<fftw_complex *>(vol.complex_data), 
				    vol.real_data, 
				    FFTW_ESTIMATE);
	backward_plans[vol.real_data] = plan;
      }
      fftw_execute(plan);
      vol *= 1.0 / vol.size;
      vol.domain = Domain::Time;
    }
  }

  void FFT::forward(VolumeList &volumes) {
    if (volumes.size() > 0 && volumes[0].domain != Domain::Frequency) {
      fftw_plan plan;
      try {
	plan = forward_plans.at(volumes.data);
      } catch (...) {
	int rank = 3;
	int n[] = { static_cast<int>(volumes[0].width), 
		    static_cast<int>(volumes[0].height),
		    static_cast<int>(volumes[0].depth) };
	int howmany = volumes.size();
	int idist = volumes.volume_size_real;
	int odist = volumes.volume_size_complex;
	int istride = 1, ostride = 1;
	int *inembed = 0, *onembed = 0;
	plan = fftw_plan_many_dft_r2c(rank, n, howmany,
				      volumes.data,
				      inembed, istride, idist,
				      reinterpret_cast<fftw_complex *>(volumes.data),
				      onembed, ostride, odist,
				      FFTW_ESTIMATE);
	forward_plans[volumes.data] = plan;
      }
      fftw_execute(plan);
      for (size_t i = 0; i < volumes.size(); ++i) {
	volumes[i].domain = Domain::Frequency;
      }
    }
  }

  void FFT::backward(VolumeList &volumes) {
    if (volumes.size() > 0 && volumes[0].domain != Domain::Time) {
      fftw_plan plan;
      try {
	plan = backward_plans.at(volumes.data);
      } catch (...) {
	int rank = 3;
	int n[] = { static_cast<int>(volumes[0].width), 
		    static_cast<int>(volumes[0].height),
		    static_cast<int>(volumes[0].depth) };
	int howmany = volumes.size();
	int idist = volumes.volume_size_complex;
	int odist = volumes.volume_size_real;
	int istride = 1, ostride = 1;
	int *inembed = 0, *onembed = 0;
	plan = fftw_plan_many_dft_c2r(rank, n, howmany,
				      reinterpret_cast<fftw_complex *>(volumes.data),
				      inembed, istride, idist,
				      volumes.data,
				      onembed, ostride, odist,
				      FFTW_ESTIMATE);
	backward_plans[volumes.data] = plan;
      }
      fftw_execute(plan);
      for (size_t i = 0; i < volumes.size(); ++i) {
	volumes[i] *= 1.0 / volumes[i].size;
	volumes[i].domain = Domain::Time;
      }
    }
  }

  void FFT::convolve(Volume &vol, Volume &mask) {
    forward(vol);
    forward(mask);
    vol *= mask;
    size_t i = 0;
    for (size_t x = 0; x < vol.width; ++x) {
      for (size_t y = 0; y < vol.height; ++y) {
    	for (size_t z = 0; z < vol.complex_depth; ++z) {
    	  if ((x+y+z) % 2) {
    	    vol.complex_data[i] *= -1.0;
    	  }
    	  ++i;
    	}
      }
    }
    backward(vol);
  }

  void FFT::convolve(VolumeList &vols, Volume &mask) {
    if (vols.size() > 0 && 
	vols[0].width == mask.width &&
	vols[0].height == mask.height &&
	vols[0].depth == mask.depth) {
      forward(vols);
      forward(mask);
      vols *= mask;
      size_t i = 0;
      for (size_t x = 0; x < vols[0].width; ++x) {
	for (size_t y = 0; y < vols[0].height; ++y) {
	  for (size_t z = 0; z < vols[0].complex_depth; ++z) {
	    if ((x+y+z) % 2) {
	      for (size_t j = 0; j < vols.size(); ++j) {
		vols[j].complex_data[i] *= -1.0;
	      }
	    }
	    ++i;
	  }
	}
      }
      backward(vols);
    }
  }
}
