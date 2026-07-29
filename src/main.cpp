#include "pathfinding/ExtractGraph.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>
// #include <vector>

int main() {
  // cv::VideoCapture cap(1);
  // if (!cap.isOpened()) {
  //   std::cerr
  //       << "Warning: Cannot open VideoCapture(1). Falling back to index 0..."
  //       << std::endl;
  //   cap.open(0);
  //   if (!cap.isOpened()) {
  //     std::cerr << "Error: Cannot open any camera!" << std::endl;
  //     return -1;
  //   }
  // }
  //
  // std::cout << "Camera opened (index 1 or 0). Press ESC or 'q' to quit."
  //           << std::endl;

  cv::VideoCapture cap("/Users/williamdolier/Documents/school/se/cpp/"
                       "pathfinding/test/rec.mp4");
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open the video file." << std::endl;
    return -1;
  }

  double fps = cap.get(cv::CAP_PROP_FPS);
  std::cout << "Playing video at " << fps << " FPS." << std::endl;

  int delay = (fps > 0) ? static_cast<int>(1000 / fps) : 33;

  namedWindow("Skeleton + Graph", cv::WINDOW_NORMAL);
  namedWindow("tracked", cv::WINDOW_NORMAL);
  namedWindow("green", cv::WINDOW_NORMAL);

  GraphExtractor graphExtractor("path.mp4");

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
    std::vector<TrackedEdge> trackedLines = graphExtractor.getTrackedEdges();
    TrackedNode *target = graphExtractor.getTarget();

    std::vector<Node> path;

    auto end = std::chrono::steady_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -
    //                                                                    start)
    //                  .count()
    //           << std::endl;

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

    if (target) {
      cv::circle(trackedNodesImg, target->pos, 3, cv::Scalar(255, 255, 0), -1);
    }

    for (const auto &edge : lines) {
      for (const auto pos : edge.path) {
        skeletonizedImageGraph.at<cv::Vec3b>(pos.y, pos.x) =
            cv::Vec3b(255, 0, 0);
      }
    }

    for (const auto &edge : trackedLines) {
      for (const auto pos : edge.path) {
        trackedNodesImg.at<cv::Vec3b>(pos.y, pos.x) = cv::Vec3b(255, 0, 0);
      }
      cv::Point src = graphExtractor.trackedGraph.nodeFromID(edge.src)->pos;
      cv::Point dst = graphExtractor.trackedGraph.nodeFromID(edge.dst)->pos;

      cv::Point srcAngle =
          src +
          static_cast<cv::Point>(20 * cv::Point2f(std::cos(edge.angleFromSrc),
                                                  std::sin(edge.angleFromSrc)));
      cv::Point dstAngle =
          dst +
          static_cast<cv::Point>(20 * cv::Point2f(std::cos(edge.angleFromDst),
                                                  std::sin(edge.angleFromDst)));

      // std::cout << edge.angleFromSrc << ", " << edge.angleFromDst << ", " <<
      // src
      //           << ", " << dst << ", " << srcAngle << ", " << dstAngle
      //           << std::endl;

      cv::line(trackedNodesImg, src, srcAngle, cv::Scalar(0, 0, 255));
      cv::line(trackedNodesImg, dst, dstAngle, cv::Scalar(255, 0, 255));
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

    graphExtractor.pathWriter.write(trackedNodesImg);

    char key = (char)cv::waitKey(delay);
    if (key == 27 || key == 'q' || key == 'Q')
      break;
  }
  // cv::waitKey(0);

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
