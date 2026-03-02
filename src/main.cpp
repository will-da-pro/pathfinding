#include "thinning/ExtractGraph.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>

int main() {
  cv::VideoCapture cap(1);
  if (!cap.isOpened()) {
    std::cerr
        << "Warning: Cannot open VideoCapture(1). Falling back to index 0..."
        << std::endl;
    cap.open(0);
    if (!cap.isOpened()) {
      std::cerr << "Error: Cannot open any camera!" << std::endl;
      return -1;
    }
  }

  std::cout << "Camera opened (index 1 or 0). Press ESC or 'q' to quit."
            << std::endl;

  namedWindow("Skeleton + Graph", cv::WINDOW_NORMAL);

  GraphExtractor graphExtractor;

  while (true) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      std::cerr << "Empty frame - exiting." << std::endl;
      break;
    }

    graphExtractor.loadImage(frame);
    graphExtractor.processImage();

    cv::Mat skeletonizedImage = graphExtractor.getSkeletonizedImage();
    cv::Mat skeletonizedImageGraph;
    cv::cvtColor(skeletonizedImage, skeletonizedImageGraph, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point> nodes = graphExtractor.getNodes();
    std::map<cv::Point, std::vector<cv::Point>, ComparePoints> lines =
        graphExtractor.getLines();

    std::vector<cv::Point> path;

    if (nodes.size() > 0) {
      path = graphExtractor.findPath(nodes[0]);
    }

    for (const auto &node : nodes) {
      cv::circle(skeletonizedImageGraph, node, 3, cv::Scalar(255, 0, 0), 5);

      for (const auto &connectedNode : lines[node]) {
        cv::line(skeletonizedImageGraph, node, connectedNode,
                 cv::Scalar(0, 255, 0), 2);
      }
    }

    for (int i = 1; i < path.size(); i++) {
      cv::line(skeletonizedImageGraph, path[i - 1], path[i],
               cv::Scalar(0, 0, 255), 3);
    }

    cv::imshow("Skeleton + Graph", skeletonizedImageGraph);

    char key = (char)cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q')
      break;
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
