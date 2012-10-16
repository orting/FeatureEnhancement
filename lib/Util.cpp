#include "Util.h"

namespace feature_enhancement {
  /* calculates eigenvalue of 3x3 symmetric matrix. source:
     https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
  */
  void calculate_eigenvalues(std::array<double, 6> const &matrix, 
			     std::array<double, 3> &eigenvalues) {
    double dxx = matrix[0];
    double dxy = matrix[1];
    double dxz = matrix[2];
    double dyy = matrix[3];
    double dyz = matrix[4];
    double dzz = matrix[5];

    double p(dxy * dxy + dxz * dxz + dyz * dyz);
    if (p == 0) {
      // The matrix is diagonal.
      eigenvalues[0] = dxx;
      eigenvalues[1] = dyy;
      eigenvalues[2] = dzz;
    }
    else {
      double q((dxx + dyy + dzz) / 3);
      p = (dxx - q)*(dxx - q) + (dyy - q)*(dyy - q) + (dzz - q) * (dzz- q) + 2 * p;
      p = sqrt(p / 6);
      /* We need to compute
       * B = (matrix - Identity * q) / p
       * r = det(B)/2
       */
      double b00 = (dxx - q) / p;
      double b11 = (dyy - q) / p;
      double b22 = (dzz - q) / p;
      double b01 = dxy / p;
      double b02 = dxz / p;
      double b12 = dyz / p;
      double r = (b00 * b11 * b22 + 2 * b01 * b02 * b12
		  - b02 * b02 * b11 - b01 * b01 * b22- b00 * b12 * b12
		  ) / 2.0;

      // In exact arithmetric for a symmetric matrix  -1 <= r <= 1
      // but computation error can leave it slightly outside this range.
      double phi;
      if (r <= -1) {
	phi = M_PI / 3;
      }
      else if (r >= 1) {
	phi = 0;
      }
      else {
	phi = acos(r) / 3;
      }
    
      // the eigenvalues satisfy eig3 <= eig2 <= eig1
      eigenvalues[0] = q + 2 * p * cos(phi);
      eigenvalues[2] = q + 2 * p * cos(phi + M_PI * (2.0/3.0));
      eigenvalues[1] = 3 * q - eigenvalues[0] - eigenvalues[2];   //  since trace(A) = eig1 + eig2 + eig3
    }
  }
}
