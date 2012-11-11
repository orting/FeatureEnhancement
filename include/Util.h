#include <array>
#include <cmath>
//#include "pechin_wrap.h"

namespace feature_enhancement {

  // struct BoundingCube {
  //   int start_x, end_x, start_y, end_y, start_z, end_z;
  // };

  // template<typename CImgType>
  // BoundingCube get_bounding_cube(cimg_library::CImg<CImgType> const &volume) {
  //   BoundingCube cube;
  //   cube.start_x = volume.width();
  //   cube.start_y = volume.height();
  //   cube.start_z = volume.depth();
  //   cube.end_x = cube.end_y = cube.end_z = 0;
  
  //   cimg_forXYZ(volume, x, y, z) {
  //     if (volume(x, y, z) != 0) {
  // 	if (x < cube.start_x) cube.start_x = x;
  // 	if (y < cube.start_y) cube.start_y = y;
  // 	if (z < cube.start_z) cube.start_z = z;
  // 	if (x > cube.end_x) cube.end_x = x;
  // 	if (y > cube.end_y) cube.end_y = y;
  // 	if (z > cube.end_z) cube.end_z = z;
  //     } 
  //   }
  //   if ((cube.end_x - cube.start_x) % 2) --cube.end_x;
  //   if ((cube.end_y - cube.start_y) % 2) --cube.end_y;
  //   if ((cube.end_z - cube.start_z) % 2) --cube.end_z;

  //   return cube;
  // }


  // template<typename CImgType>
  // void center_fft(cimg_library::CImg<CImgType> &vol) {
  //   for (int z = 0; z < vol.depth(); ++z) {
  //     for (int y = 0; y < vol.height(); ++y) {
  // 	for (int x = 0, i = 0; x < vol.width(); x+=2, ++i) {
  // 	  if ((i+z+y) % 2) {
  // 	    vol(x,y,z) *=  -1;
  // 	    vol(x+1,y,z) *= -1;
  // 	  }
  // 	}
  //     }
  //   }
  // }

  // template<typename CImgType>
  // void insert_at(cimg_library::CImg<CImgType> const &src, 
  // 		 cimg_library::CImg<CImgType> &dst, 
  // 		 BoundingCube const &cube) {
  //   for (int x = cube.start_x, i = 0; x < cube.end_x; ++x, ++i) {
  //     for (int y = cube.start_y, j = 0; y < cube.end_y; ++y, ++j) {
  // 	for (int z = cube.start_z, k = 0; z < cube.end_z; ++z, ++k) {
  // 	  dst(x, y, z) = src(i, j, k);
  // 	}
  //     }
  //   }
  // }


  template<typename NumType>
  NumType calculate_gradient(NumType dx, NumType dy, NumType dz) {
    return std::sqrt(dx*dx + dy*dy + dz*dz);
  }


  void calculate_eigenvalues(std::array<double, 6> const &matrix, 
			     std::array<double, 3> &eigenvalues);

}
