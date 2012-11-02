#include <math.h>
#include "FeatureMeasure.h"

namespace feature_enhancement {

  /* Calculates a fissureness measure using the method described in
   * Rikxoort et al: Supervised Enhancement Filters: Application to Fissure Detection in Chest CT Scans
   */
  double fissureness_rikxoort(double const hounsfield_mean, double const hounsfield_sd, 
			      double voxel_value, double eig1, double eig2, double eig3) {
    // we need the numericaly largest of the other 2 eigenvalues
    double abs_eig1 = fabs(eig1);
    double abs_eig2 = fabs(eig2);
    double abs_eig3 = fabs(eig3);
    if (abs_eig1 > abs_eig2) {
      abs_eig2 = abs_eig1;
    }

    double plateness = (abs_eig3 - abs_eig2) / (abs_eig3 + abs_eig2);

    double hounsfield_diff = voxel_value - hounsfield_mean;
    double hounsfield = exp(- (hounsfield_diff * hounsfield_diff) / (2 * hounsfield_sd * hounsfield_sd));

    return plateness * hounsfield;
  }

  /* Calculates a fissure similarity meassure using the method described in
   * Lassen et al: AUTOMATIC SEGMENTATION OF LUNG LOBES IN CT IMAGES BASED ON FISSURES, VESSELS, AND BRONCHI
   * requires: eig1 >= eig2 >= eig3
    // alpha, beta, gamma are empirical, see Lassen et al
   */
  double fissureness_lassen(double const alpha, double const beta, double const gamma,
			    double eig1, double eig2, double eig3) {
    if (eig3 > 0) {
      return 0;
    }
    double const beta6 = beta * beta * beta * beta * beta * beta;
    double const gamma6 = gamma * gamma * gamma * gamma * gamma * gamma;

    // we need to use the numericaly largest of the other eigenvalues
    if (fabs(eig1) > fabs(eig2)) {
      eig2 = eig1;
    }

    // double structure = exp(- pow(eig3 - alpha, 6) / beta6); From the paper, but makes no sense
    // since it gives higher structure value to dark voxels with light neighbours
    double structure = exp(- pow(eig3 + alpha, 6) / beta6);
    double sheet = exp(- pow(eig2, 6) / gamma6);

    double fissure = structure * sheet;

    return fissure;
  }
}
