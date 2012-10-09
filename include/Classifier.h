#include <vector>
#include <string>

namespace feature_enhancement {
  class Classifier {
  public:
    Classifier();
    Classifier(std::vector< std::vector<short> > &classification_matrix);
    ~Classifier(){};
    
    void load(std::string filename);
    void save(std::string filename);

    void set_classification_matrix(std::vector< std::vector<short> > &classification_matrix);
    
    std::vector<int> knn(std::vector<short> features, int k);
    std::vector< std::vector<int> > knn(std::vector< std::vector<short> > features, int k);
    
  private:
    int distance(std::vector<short> p1, std::vector<short> p2);
    std::vector< std::vector<short> > classifications;
  };

}
