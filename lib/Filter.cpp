#include <iostream>
#include "Filter.h"

namespace feature_enhancement {
  void kernel(Filter3D filter, int scale, Volume &out) {
    size_t w = static_cast<size_t>(6 * scale + 1);
    size_t k_w = w > out.width ? out.width : w;
    size_t k_h = w > out.height ? out.height : w;
    size_t k_d = w > out.depth ? out.depth : w;
    size_t half_w = out.width / 2;
    size_t half_h = out.height / 2;
    size_t half_d = out.depth / 2;
    size_t start_i = half_w - k_w / 2;
    size_t start_j = half_h - k_h / 2;
    size_t start_k = half_d - k_d / 2;
    size_t end_i = start_i + k_w;
    size_t end_j = start_j + k_h;
    size_t end_k = start_k + k_d;

    double x = - static_cast<double>(k_w/2);
    double start_y = - static_cast<double>(k_h/2);
    double start_z = - static_cast<double>(k_d/2);
    for (size_t i = start_i; i < end_i; ++i, ++x) {
      double y = start_y;
      for (size_t j = start_j; j < end_j; ++j, ++y) {
	double z = start_z;
	for (size_t k = start_k; k < end_k; ++k, ++z) {
	  out(i, j, k) = filter(x, y, z, scale);
	}
      }
    }
  }
}
