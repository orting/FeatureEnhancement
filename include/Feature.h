#pragma once

namespace feature_enhancement {
  enum Feature {
    Identity,
    Gauss,
    GaussDx,
    GaussDy,
    GaussDz,
    GaussDxx,
    GaussDxy,
    GaussDxz,
    GaussDyy,
    GaussDyz,
    GaussDzz,
    Gradient,
    HessianEig1,
    HessianEig2,
    HessianEig3,
    OutOfBounds
  };

  Feature& operator++(Feature& orig);
  Feature operator++(Feature& orig, int);
}
