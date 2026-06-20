#include "pathfinding/ExtractGraph.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>
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
  namedWindow("tracked", cv::WINDOW_NORMAL);
  namedWindow("green", cv::WINDOW_NORMAL);

  GraphExtractor graphExtractor;

  while (true) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      std::cerr << "Empty frame - exiting." << std::endl;
      break;
    }

    // cv::Mat frame = cv::imread(
    //     "/Users/williamdolier/Documents/school/se/cpp/pathfinding/line.jpg",
    //     cv::IMREAD_COLOR);

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
    std::vector<TrackedNode> trackedNodes = graphExtractor.getTrackedNodes();

    std::vector<Node> path;

    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << std::endl;

    if (nodes.size() > 0) {
      path = graphExtractor.findPath(nodes[0]);
    }

    cv::Mat trackedNodesImg = skeletonizedImageGraph.clone();

    for (const auto &node : trackedNodes) {
      cv::Scalar color = cv::Scalar(255, 0, 0);

      if (node.screen_edge)
        color = cv::Scalar(0, 255, 0);

      cv::circle(trackedNodesImg, node.pos, 3, color, -1);
    }

    for (const auto &node : nodes) {
      cv::Scalar color = cv::Scalar(255, 0, 0);

      if (node.screen_edge)
        color = cv::Scalar(0, 255, 0);

      cv::circle(skeletonizedImageGraph, node.pos, 3, color, -1);
    }

    for (const auto &edge : lines) {
      for (const auto pos : edge.path) {
        skeletonizedImageGraph.at<cv::Vec3b>(pos.y, pos.x) =
            cv::Vec3b(255, 0, 0);
      }
    }

    for (int i = 1; i < path.size(); i++) {
      cv::line(skeletonizedImageGraph, path[i - 1].pos, path[i].pos,
               cv::Scalar(0, 0, 255), 3);
    }

    std::vector<cv::Point> greenCenters = graphExtractor.extractGreen(frame);
    cv::Mat green = frame.clone();

    for (const auto &center : greenCenters) {
      cv::circle(green, center, 25, cv::Scalar(0, 0, 255), -1);

      cv::Point newCenter =
          graphExtractor.cvtPoint(green, skeletonizedImageGraph, center);
      cv::circle(skeletonizedImageGraph, newCenter, 5, cv::Scalar(0, 255, 255),
                 -1);
    }

    cv::imshow("Skeleton + Graph", skeletonizedImageGraph);
    cv::imshow("tracked", trackedNodesImg);
    cv::imshow("green", green);

    char key = (char)cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q')
      break;
  }
  // cv::waitKey(0);

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
