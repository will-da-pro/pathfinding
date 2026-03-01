#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>

// Simple structure to hold graph data
struct Node {
  cv::Point position;
  std::vector<int> neighbors; // Indices of connected nodes
};

void followPath(const cv::Mat &skel, std::vector<Node> &map,
                cv::Point currentPos, cv::Point lastPos, int lastNode);
void linkNode(const cv::Mat &skel, std::vector<Node> &map, int node);

// Function to check if a pixel is a branch/junction or endpoint
bool isNode(const cv::Mat &skel, int x, int y) {
  int neighbors = 0;
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      if (i == 0 && j == 0)
        continue;
      if (skel.at<uchar>(y + i, x + j) == 255)
        neighbors++;
    }
  }
  // Endpoints (1) and Junctions (>2) are nodes
  return neighbors == 1 || neighbors > 2;
}

std::vector<cv::Point> surroundingPoints(const cv::Mat &skel, cv::Point pos) {
  std::vector<cv::Point> points = {};
  // std::cout << points << std::endl;

  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      // std::cout << pos.x + j << ", " << pos.y + i << "xy\n";
      if (i == 0 && j == 0)
        continue;
      if ((pos.y + i < 0) || (pos.y + i >= skel.rows) || (pos.x + j < 0) ||
          (pos.x + j >= skel.cols))
        continue;

      // std::cout << "yes\n";

      if (skel.at<uchar>(pos.y + i, pos.x + j) == 255)
        points.push_back(cv::Point(pos.x + j, pos.y + i));

      // std::cout << cv::Point(pos.x + j, pos.y + i) << std::endl;
      // std::cout << points << std::endl;
    }
  }

  // std::cout << "got points\n";
  // std::cout << "Size: " << points.size() << std::endl;
  // std::cout << points << std::endl;
  // std::cout << "printed points\n";

  return points;
}

void followPath(const cv::Mat &skel, std::vector<Node> &map,
                cv::Point currentPos, cv::Point lastPos, int lastNode) {
  std::cout << "Getting surrounding\n";
  std::vector<cv::Point> surrounding = surroundingPoints(skel, currentPos);
  std::cout << "Got surrounding " << surrounding << std::endl;

  for (const auto &point : surrounding) {
    if (point == lastPos)
      continue;

    bool isNodePos = false;
    int nodeIndex;

    for (nodeIndex = 0; nodeIndex < map.size(); nodeIndex++) {
      if (map[nodeIndex].position == currentPos) {
        isNodePos = true;
        break;
      }
    }

    std::cout << "Node index: " << nodeIndex << isNodePos << std::endl;

    if (isNodePos) {
      map[lastNode].neighbors.push_back(nodeIndex);
      if (map[nodeIndex].neighbors.size() == 0) {
        linkNode(skel, map, nodeIndex);
      }
    }

    else {
      followPath(skel, map, point, currentPos, lastNode);
    }
  }
}

void linkNode(const cv::Mat &skel, std::vector<Node> &map, int node) {
  std::cout << "Node: " << node << std::endl;
  std::vector<cv::Point> surrounding =
      surroundingPoints(skel, map[node].position);

  for (const auto &pos : surrounding) {
    std::cout << "pos: " << map[node].position << ", surrounding pos: " << pos
              << std::endl;
    followPath(skel, map, pos, map[node].position, node);
  }
}

int main(int argc, char **argv) {
  cv::VideoCapture cap(1);

  if (!cap.isOpened()) {
    std::cerr << "Cap not opened!\n";
    return -1;
  }

  cv::namedWindow("Video Stream", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Grayscale", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Binary", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Skeleton", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Graph Nodes", cv::WINDOW_AUTOSIZE);

  while (true) {
    cv::Mat frame;
    bool success = cap.read(frame);

    if (!success || frame.empty()) {
      std::cerr << "Could not read frame!\n";
      break;
    }

    cv::Mat resized;
    cv::Size dsize(200, 100);

    cv::resize(frame, resized, dsize, 0, 0, cv::INTER_LINEAR);

    cv::Mat gray_image;
    cv::cvtColor(resized, gray_image, cv::COLOR_BGR2GRAY);

    cv::Mat binary_image;
    double thresh_value = 100;
    double max_value = 255;

    cv::threshold(gray_image, binary_image, thresh_value, max_value,
                  cv::THRESH_BINARY_INV);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat opened_image;
    cv::morphologyEx(binary_image, opened_image, cv::MORPH_OPEN, kernel);

    cv::Mat closed_image;
    cv::morphologyEx(opened_image, closed_image, cv::MORPH_CLOSE, kernel);

    cv::Mat skel;
    cv::ximgproc::thinning(closed_image, skel, cv::ximgproc::THINNING_GUOHALL);

    std::cout << "graphing\n";

    std::vector<Node> graph;

    // 2. Simple traversal to find nodes (endpoints/junctions)
    for (int y = 1; y < skel.rows - 1; ++y) {
      for (int x = 1; x < skel.cols - 1; ++x) {
        if (skel.at<uchar>(y, x) == 255) {
          if (isNode(skel, x, y)) {
            Node n;
            n.position = cv::Point(x, y);
            graph.push_back(n);
          }
        }
      }
    }

    std::cout << graph.size() << std::endl;

    for (int i = 0; i < graph.size(); i++) {
      std::cout << "i: " << i << std::endl;
      if (graph[i].neighbors.size() != 0)
        continue;

      linkNode(skel, graph, i);
    }

    std::cout << "linked\n";

    // 3. Visualization
    cv::Mat result;
    cv::cvtColor(skel, result, cv::COLOR_GRAY2BGR);
    for (const auto &node : graph) {
      cv::Scalar color(0, 0, 255);

      int nc = node.neighbors.size();

      if (nc == 1) {
        color = cv::Scalar(255, 0, 0);
      }

      else if (nc == 3) {
        color = cv::Scalar(0, 255, 0);
      }

      cv::circle(result, node.position, 3, color, -1);
    }

    cv::imshow("Graph Nodes", result);

    cv::imshow("Video Stream", frame);
    cv::imshow("Grayscale", gray_image);
    cv::imshow("Binary", closed_image);
    cv::imshow("Skeleton", skel);

    if (cv::waitKey(1) == 27) {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}
