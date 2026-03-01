#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

class GraphExtractor {
public:
  GraphExtractor();
  void loadImage(cv::Mat image);
  void processImage();
  std::vector<cv::Point> getNodes();
  std::vector<std::pair<cv::Point, cv::Point>> getLines();
  cv::Mat getSkeletonizedImage();

private:
  void extractNodes();
  void extractLines();
  std::vector<cv::Point> getSurroundingPoints(cv::Point centre, int radius);

  cv::Mat rawImage;
  cv::Mat skeletonizedImage;
  std::vector<cv::Point> nodes;
  std::vector<std::pair<cv::Point, cv::Point>> lines;
};
