#include <algorithm>
#include <iostream>
#include <fstream>
#include "Classifier.h"

using namespace feature_enhancement;

Classifier::Classifier():
  classifications() {
}

Classifier::Classifier(std::vector< std::vector<short> > &classification_matrix):
  classifications(classification_matrix) {
  
}

void Classifier::load(std::string filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  
  this->classifications.clear();
  std::vector<short> features;
  short feature;
  while(in.good()) {
    while (in.peek() == ' ') {
      in.get();
    }
    if (in.peek() == '\n') {
      in.get();
      this->classifications.push_back(features);
      features.clear();
    }
    else {
      in >> feature;
      features.push_back(feature);
    }
  }
}

void Classifier::save(std::string filename) {
  std::ofstream out;
  out.open(filename, std::ofstream::out);

  for (auto p : this->classifications) {
    for (auto f : p) {
      out << f << " ";
    }
    out << '\n';
  }
}

std::vector<int> Classifier::knn(std::vector<short> point, int k) {
  std::vector< std::pair<int,int> > distances;
  distances.reserve(this->classifications.size());

  for (auto p: this->classifications) {
    distances.push_back(std::make_pair(distance(point, p), p.back()));
  }

  std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

  std::vector<int> nn;
  nn.resize(k);
  for (int i = 0; i < k; ++i) {
    nn[i] = distances[i].second;
  }

  return nn;
}


std::vector< std::vector<int> > Classifier::knn(std::vector< std::vector<short> > points, int k) {
  std::vector< std::vector<int> > nns;
  nns.reserve(points.size());

  for (auto p: points) {
    nns.push_back(this->knn(p, k));
  }
  
  return nns;
}


int Classifier::distance(std::vector<short> p1, std::vector<short> p2) {
  int d = 0;
  for (size_t i = 0; i < std::min(p1.size(), p2.size()); ++i) {
    int diff = p1[i] - p2[i];
    d += diff * diff;
  }
  return d;
}


void Classifier::set_classification_matrix(std::vector< std::vector<short> > &classification_matrix) {
  this->classifications = classification_matrix;

}
