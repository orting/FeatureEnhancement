#pragma once
#include <functional>

namespace feature_enhancement {
  typedef std::function<double (double, double, double, double)> FeatureMeassure;

  double fissureness_lassen(double const alpha, double const beta, double const gamma,
			    double eig1, double eig2, double eig3);
  double fissureness_rikxoort(double hounsfield_mean, double hounsfield_sd, 
			      double voxel_value, double eig1, double eig2, double eig3);
}
