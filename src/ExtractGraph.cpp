#include "thinning/ExtractGraph.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/ximgproc.hpp>
#include <stdexcept>
#include <vector>

GraphExtractor::GraphExtractor() {
  std::cout << "Initialising graph extractor..." << std::endl;
  this->pathLimit = 5;
}

void GraphExtractor::loadImage(cv::Mat image) {
  cv::Mat resized;
  cv::Size dsize(200, 100);

  cv::resize(image, resized, dsize, 0, 0, cv::INTER_LINEAR);

  cv::Mat gray, binary;
  cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);
  threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

  cv::Mat opened_image;
  cv::morphologyEx(binary, opened_image, cv::MORPH_OPEN, kernel);

  cv::Mat closed_image;
  cv::morphologyEx(opened_image, closed_image, cv::MORPH_CLOSE, kernel);

  cv::Mat skeleton;
  cv::ximgproc::thinning(closed_image, skeleton,
                         cv::ximgproc::THINNING_GUOHALL);

  int rows = skeleton.rows;
  int cols = skeleton.cols;

  cv::Rect top_border_roi(0, 0, cols, 1);
  skeleton(top_border_roi).setTo(0);

  // Set bottom border
  cv::Rect bottom_border_roi(0, rows - 1, cols, 1);
  skeleton(bottom_border_roi).setTo(0);

  // Set left border (excluding corners already set)
  cv::Rect left_border_roi(0, 1, 1, rows - 2);
  skeleton(left_border_roi).setTo(0);

  // Set right border (excluding corners already set)
  cv::Rect right_border_roi(cols - 1, 1, 1, rows - 2);
  skeleton(right_border_roi).setTo(0);

  this->rawImage = image;
  this->skeletonizedImage = skeleton;
}

cv::Mat GraphExtractor::getSkeletonizedImage() {
  return this->skeletonizedImage;
}

void GraphExtractor::processImage() {
  if (this->skeletonizedImage.empty()) {
    std::cerr
        << "Attempting to process empty image! Image must be loaded first."
        << std::endl;
    return;
  }
  this->extractNodes();
  this->extractLines();
}

std::vector<cv::Point> GraphExtractor::getNodes() { return this->nodes; }

std::map<cv::Point, std::vector<cv::Point>, ComparePoints>
GraphExtractor::getLines() {
  return this->lines;
}

void GraphExtractor::extractNodes() {
  cv::Mat image = this->skeletonizedImage;
  std::vector<cv::Point> whitePixels;
  std::vector<cv::Point> foundNodes;

  cv::findNonZero(image, whitePixels);

  for (const auto &point : whitePixels) {
    std::vector<cv::Point> surroundingPoints =
        this->getSurroundingPoints(point, 3);

    if (surroundingPoints.size() != 3) {
      foundNodes.push_back(point);
    }
  }

  this->nodes = foundNodes;
}

std::vector<cv::Point> GraphExtractor::getSurroundingPoints(cv::Point centre,
                                                            int radius) {
  cv::Mat image = this->skeletonizedImage;
  cv::Rect roi(centre.x - 1, centre.y - 1, radius, radius);
  std::vector<cv::Point> surroundingPoints;

  if (centre.x <= 0 || centre.y <= 0 || centre.x >= image.cols - 1 ||
      centre.y >= image.rows - 1) {
    return surroundingPoints;
  }

  cv::Mat cropped;
  cropped = image(roi).clone();

  cv::findNonZero(cropped, surroundingPoints);

  for (auto &point : surroundingPoints) {
    point += centre + cv::Point(-1, -1);
  }

  return surroundingPoints;
}

void GraphExtractor::extractLines() {
  if (this->nodes.size() == 0) {
    return;
  }

  for (const auto &node : this->nodes) {
    this->lines[node] = {};
    std::vector<cv::Point> connectedNodes = this->getConnectedNodes(node);

    for (const auto &connected : connectedNodes) {
      this->lines[node].push_back(connected);
    }
  }
}

std::vector<cv::Point> GraphExtractor::getConnectedNodes(cv::Point node) {
  std::vector<cv::Point> connectedNodes;
  std::vector<cv::Point> surroundingPoints =
      this->getSurroundingPoints(node, 3);

  for (const auto &point : surroundingPoints) {
    if (point == node)
      continue;

    cv::Point connectedNode = this->followToNode(point, node);
    connectedNodes.push_back(connectedNode);
  }

  return connectedNodes;
}

double GraphExtractor::calculateAngle(cv::Point point1, cv::Point point2) {
  int rise = point2.y - point1.y;
  int run = point2.x - point1.x;

  double angle = std::atan2(rise, run);
  return angle;
}

cv::Point GraphExtractor::followToNode(cv::Point current, cv::Point previous) {
  if (std::find(this->nodes.begin(), this->nodes.end(), current) !=
      this->nodes.end()) {
    return current;
  }

  std::vector<cv::Point> surroundingPoints =
      this->getSurroundingPoints(current, 3);

  auto it1 =
      std::find(surroundingPoints.begin(), surroundingPoints.end(), current);

  if (it1 != surroundingPoints.end()) {
    surroundingPoints.erase(it1);
  }

  auto it2 =
      std::find(surroundingPoints.begin(), surroundingPoints.end(), previous);

  if (it2 != surroundingPoints.end()) {
    surroundingPoints.erase(it2);
  }

  if (surroundingPoints.size() != 1) {
    throw std::runtime_error("Line does not end in node!");
  }

  return this->followToNode(surroundingPoints[0], current);
}

void GraphExtractor::findNextNode(std::vector<cv::Point> &path) {
  cv::Point current = path[path.size() - 1];
  cv::Point previous = path[path.size() - 2];
  std::vector<cv::Point> connectedNodes = this->lines[current];

  auto it1 = std::find(connectedNodes.begin(), connectedNodes.end(), current);

  if (it1 != connectedNodes.end()) {
    connectedNodes.erase(it1);
  }

  auto it2 = std::find(connectedNodes.begin(), connectedNodes.end(), previous);

  if (it2 != connectedNodes.end()) {
    connectedNodes.erase(it2);
  }

  if (connectedNodes.size() == 0 || path.size() > this->pathLimit) {
    return;
  }

  double previousAngle = this->calculateAngle(current, previous);
  double targetAngle = fmod(previousAngle + M_PI, M_PI);
  double closestAngle = this->calculateAngle(current, connectedNodes[0]);
  cv::Point closestNode = connectedNodes[0];

  for (const auto &node : connectedNodes) {
    double angle = this->calculateAngle(current, node);
    if (abs(angle - targetAngle) < abs(closestAngle - targetAngle)) {
      closestAngle = angle;
      closestNode = node;
    }
  }

  path.push_back(closestNode);
  this->findNextNode(path);
}

std::vector<cv::Point> GraphExtractor::findPath(cv::Point startPos) {
  std::vector<cv::Point> path;

  // TODO find nearest node instead of only using exact position
  std::vector<cv::Point> connectedNodes = this->lines[startPos];

  if (connectedNodes.size() > 0) {
    path.push_back(startPos);
    path.push_back(connectedNodes[0]);

    this->findNextNode(path);
  }

  return path;
}
