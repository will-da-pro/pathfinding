#include "pathfinding/ExtractGraph.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
// #include <vector>

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
    auto start = std::chrono::steady_clock::now();
    // Graph graph = buildGraph(skeletonizedImage, 10);

    // cv::Mat skeletonizedImageGraph = visualise(skeletonizedImage, graph);

    cv::Mat skeletonizedImageGraph;
    cv::cvtColor(skeletonizedImage, skeletonizedImageGraph, cv::COLOR_GRAY2BGR);

    std::vector<Node> nodes = graphExtractor.getNodes();
    std::vector<Edge> lines = graphExtractor.getEdges();

    std::vector<cv::Point> path;

    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << std::endl;

    if (nodes.size() > 0) {
      path = graphExtractor.findPath(nodes[0]);
    }

    for (const auto &node : nodes) {
      cv::circle(skeletonizedImageGraph, node.pos, 3, cv::Scalar(255, 0, 0), 5);
    }

    for (const auto &edge : lines) {
      for (const auto pos : edge.path) {
        skeletonizedImageGraph.at<cv::Vec3b>(pos.y, pos.x) =
            cv::Vec3b(255, 0, 0);
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
