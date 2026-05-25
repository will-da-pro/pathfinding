#include "pathfinding/ExtractGraph.hpp"
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
  this->minEdgeSize = 15;
}

void GraphExtractor::loadImage(cv::Mat image) {
  cv::Mat resized;
  cv::Size dsize(200, 100);

  cv::resize(image, resized, dsize, 0, 0, cv::INTER_LINEAR);

  cv::Mat gray, binary;
  cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);

  cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY_INV, 11, 2);

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

Node *GraphExtractor::nodeFromID(int id) {
  for (Node &node : this->graph.nodes) {
    if (node.id == id)
      return &node;
  }

  return nullptr;
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
  this->extractEdges();
  this->removeShortEdges(this->graph.edges);
}

std::vector<Node> GraphExtractor::getNodes() { return this->graph.nodes; }

std::vector<Edge> GraphExtractor::getEdges() { return this->graph.edges; }

void GraphExtractor::extractNodes() {
  cv::Mat image = this->skeletonizedImage;
  std::vector<cv::Point> whitePixels;
  std::vector<Node> foundNodes;

  cv::findNonZero(image, whitePixels);

  for (const auto &point : whitePixels) {
    std::vector<cv::Point> surroundingPoints =
        this->getSurroundingPoints(point, 3);

    if (surroundingPoints.size() == 3)
      continue;

    Node node;
    node.pos = point;
    node.id = foundNodes.size();

    if (surroundingPoints.size() > 3) {
      node.is_endpoint = false;
    } else {
      node.is_endpoint = true;
    }

    foundNodes.push_back(node);
  }

  this->graph.nodes = foundNodes;
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

void GraphExtractor::extractEdges() {
  if (this->graph.nodes.size() == 0) {
    return;
  }

  std::vector<Edge> edges;

  for (const auto &node : this->graph.nodes) {
    // unoptimised—Should check if node path exists on edge before tracing
    std::vector<Edge> connectedEdges = this->traceConnectedEdges(node);

    for (const auto &edge : connectedEdges) {
      bool exists = false;

      for (const auto &existingEdge : edges) {
        if (edge == existingEdge) {
          exists = true;
          break;
        }
      }

      if (!exists) {
        edges.push_back(edge);
      }
    }
  }

  this->graph.edges = edges;
}

std::vector<Edge *> GraphExtractor::getConnectedEdges(int nodeID) {
  std::vector<Edge *> result;

  for (Edge &edge : this->graph.edges) {
    if (edge.src == nodeID || edge.dst == nodeID)
      result.push_back(&edge);
  }

  return result;
}

void GraphExtractor::removeShortEdges(std::vector<Edge> &edges) {
  for (int i = 0; i < edges.size(); i++) {
    // If the edge is long enough, do nothing.
    if (edges[i].path.size() >= this->minEdgeSize)
      continue;

    Node *src = this->nodeFromID(edges[i].src);
    Node *dst = this->nodeFromID(edges[i].dst);

    if (!src || !dst) {
      std::cerr << "Node does not exist!\n";
      continue;
    }

    // If either of the ends of an edge are endpoints, delete it.
    if (src->is_endpoint || dst->is_endpoint) {
      edges.erase(edges.begin() + i);
      i--;
      continue;
    }

    // Merge close intersections
    for (Edge *connectedEdge : this->getConnectedEdges(edges[i].src)) {
      if (!connectedEdge) {
        std::cerr << "Edge does not exist!\n";
        continue;
      }

      *connectedEdge = this->mergeEdges(*connectedEdge, edges[i]);
    }

    edges.erase(edges.begin() + i);
    i--;
  }
}

Edge GraphExtractor::mergeEdges(Edge edge1, Edge edge2) {
  if (edge1.dst == edge2.src) {
    edge1.path.insert(edge1.path.end(), edge2.path.begin() + 1,
                      edge2.path.end());
    edge1.dst = edge2.dst;
  }

  else if (edge1.src == edge2.dst) {
    edge1.path.insert(edge1.path.begin(), edge2.path.begin() + 1,
                      edge2.path.end());
    edge1.src = edge2.src;
  }

  else if (edge1.dst == edge2.dst) {
    std::reverse(edge2.path.begin(), edge2.path.end());

    edge1.path.insert(edge1.path.end(), edge2.path.begin() + 1,
                      edge2.path.end());
    edge1.dst = edge2.src;
  }

  else if (edge1.src == edge2.src) {
    std::reverse(edge2.path.begin(), edge2.path.end());
    edge1.path.insert(edge1.path.begin(), edge2.path.begin() + 1,
                      edge2.path.end());
    edge1.src = edge2.dst;
  }

  edge1.length = edge1.path.size();
  return edge1;
}

std::vector<Edge> GraphExtractor::traceConnectedEdges(Node node) {
  std::vector<Edge> connectedEdges;
  std::vector<cv::Point> surroundingPoints =
      this->getSurroundingPoints(node.pos, 3);

  for (const auto &point : surroundingPoints) {
    if (point == node.pos)
      continue;

    Edge edge;
    edge.src = node.id;

    edge.path.push_back(node.pos);
    edge.path.push_back(point);

    edge.dst = this->followToNode(edge.path).id;
    edge.length = edge.path.size();

    connectedEdges.push_back(edge);
  }

  return connectedEdges;
}

double GraphExtractor::calculateAngle(cv::Point point1, cv::Point point2) {
  int rise = point2.y - point1.y;
  int run = point2.x - point1.x;

  double angle = std::atan2(rise, run);
  return angle;
}

Node GraphExtractor::followToNode(std::vector<cv::Point> &path) {
  cv::Point current = path[path.size() - 1];
  cv::Point previous;

  if (path.size() > 1) {
    previous = path[path.size() - 2];
  }

  auto it =
      std::find_if(this->graph.nodes.begin(), this->graph.nodes.end(),
                   [current](const Node &node) { return node.pos == current; });

  if (it != this->graph.nodes.end()) {
    return *it;
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

  path.push_back(surroundingPoints[0]);

  return this->followToNode(path);
}

void GraphExtractor::findNextNode(std::vector<Node> &path) {
  // cv::Point current = path[path.size() - 1];
  // cv::Point previous = path[path.size() - 2];
  // std::vector<cv::Point> connectedNodes = this->lines[current];
  //
  // auto it1 = std::find(connectedNodes.begin(), connectedNodes.end(),
  // current);
  //
  // if (it1 != connectedNodes.end()) {
  //   connectedNodes.erase(it1);
  // }
  //
  // auto it2 = std::find(connectedNodes.begin(), connectedNodes.end(),
  // previous);
  //
  // if (it2 != connectedNodes.end()) {
  //   connectedNodes.erase(it2);
  // }
  //
  // if (connectedNodes.size() == 0 || path.size() > this->pathLimit) {
  //   return;
  // }
  //
  // double previousAngle = this->calculateAngle(current, previous);
  // double targetAngle = fmod(previousAngle + M_PI, M_PI);
  // double closestAngle = this->calculateAngle(current, connectedNodes[0]);
  // cv::Point closestNode = connectedNodes[0];
  //
  // for (const auto &node : connectedNodes) {
  //   double angle = this->calculateAngle(current, node);
  //   if (abs(angle - targetAngle) < abs(closestAngle - targetAngle)) {
  //     closestAngle = angle;
  //     closestNode = node;
  //   }
  // }
  //
  // path.push_back(closestNode);
  // this->findNextNode(path);
}

std::vector<cv::Point> GraphExtractor::findPath(Node startPos) {
  std::vector<cv::Point> path;

  // TODO find nearest node instead of only using exact position
  // std::vector<cv::Point> connectedNodes = this->lines[startPos];
  //
  // if (connectedNodes.size() > 0) {
  //   path.push_back(startPos);
  //   path.push_back(connectedNodes[0]);
  //
  //   this->findNextNode(path);
  // }

  return path;
}
