#include <iostream>
#include <fstream>
#include "pechin_wrap.h"

#include "Filter.h"
#include "Gauss.h"

struct FilterMapping {
  std::string name;
  double (*filter)(double, double, double, int);
};

int main(int argc, char *argv[]) {
  cimg_usage("Apply a filter to the volume");
  const char* infile = cimg_option("-i", (char*)0, "Input volume file. Expects 16-bit int values");
  const char* filter = cimg_option("-f", "gauss", "Name of filter: gauss, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz");
  const int scale    = cimg_option("-s", 1, "Scale of filter: 1+");

  if (infile == 0 || scale < 1) {
    std::cerr << "Not enough arguments. Use -h to get help" << std::endl;
    return -1;
  }
  cimg_library::CImg<short> volume(infile);

  FilterMapping filters[] = { {"gauss", &gauss::gauss},
			      {"dx", &gauss::dx},
			      {"dy", &gauss::dy},
			      {"dz", &gauss::dz},
			      {"dxx", &gauss::dxx},
			      {"dxy", &gauss::dxy},
			      {"dxz", &gauss::dxz},
			      {"dyy", &gauss::dyy},
			      {"dyz", &gauss::dyz},
			      {"dzz", &gauss::dzz}};
  
  for (auto f : filters) {
    if (f.name.compare(filter) == 0) {
      filter::apply(volume, scale, *(f.filter));
      break;
    }
  }
  volume.display();

  return 0;
}
