#include <map>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

struct ComparePoints {
  bool operator()(const cv::Point &a, const cv::Point &b) const {
    if (a.y != b.y) {
      return a.y < b.y;
    }
    return a.x < b.x;
  }
};

class GraphExtractor {
public:
  GraphExtractor();
  void loadImage(cv::Mat image);
  void processImage();
  std::vector<cv::Point> getNodes();
  std::map<cv::Point, std::vector<cv::Point>, ComparePoints> getLines();
  cv::Mat getSkeletonizedImage();
  std::vector<cv::Point> findPath(cv::Point startPos);

  int pathLimit;

private:
  void extractNodes();
  void extractLines();
  std::vector<cv::Point> getSurroundingPoints(cv::Point centre, int radius);
  std::vector<cv::Point> getConnectedNodes(cv::Point node);
  cv::Point followToNode(cv::Point current, cv::Point previous);
  void findNextNode(std::vector<cv::Point> &path);
  double calculateAngle(cv::Point point1, cv::Point point2);

  cv::Mat rawImage;
  cv::Mat skeletonizedImage;
  std::vector<cv::Point> nodes;
  std::map<cv::Point, std::vector<cv::Point>, ComparePoints> lines;
};
