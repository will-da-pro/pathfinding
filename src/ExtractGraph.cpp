#include "pathfinding/ExtractGraph.hpp"
#include "Hungarian.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/ximgproc.hpp>
#include <stdexcept>
#include <vector>

TrackedNode::TrackedNode(cv::Point pos) {
  float dt = 1.0f;
  kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0,
                         0, 1, 0, 0, 0, 0, 1);

  kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);

  cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));

  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));

  cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

  kf.statePost.at<float>(1) = pos.x; // Initial X
  kf.statePost.at<float>(1) = pos.y; // Initial Y
  kf.statePost.at<float>(2) = 0.0f;  // Initial velocity X
  kf.statePost.at<float>(3) = 0.0f;  // Initial velocity Y

  this->pos = pos;
}

std::vector<std::vector<double>> TrackedGraph::getCostMatrix(Graph &graph) {
  // TODO apply kalman filter

  // The vector must be square
  int size = this->nodes.size() > graph.nodes.size() ? this->nodes.size()
                                                     : graph.nodes.size();

  std::vector<std::vector<double>> costs(size, std::vector(size, 0.0));

  for (int i = 0; i < this->nodes.size(); i++) {
    for (int j = 0; j < graph.nodes.size(); j++) {
      Node &newNode = graph.nodes[j];
      Node &trackedNode = this->nodes[i];

      double distance = cv::norm(trackedNode.pos - newNode.pos);
      int connectedEdgeDiff = graph.getConnectedEdges(newNode.id).size() -
                              this->getConnectedEdges(trackedNode.id).size();

      double penalty =
          trackedNode.screen_edge ? 0 : 0.0 * std::abs(connectedEdgeDiff);

      double cost = distance + penalty;
      costs[i][j] = penalty;
    }
  }

  return costs;
}

GraphExtractor::GraphExtractor() {
  std::cout << "Initialising graph extractor..." << std::endl;
  this->pathLimit = 5;
  this->minEdgeSize = 15;
  this->gatingThreshold = 10;
}

void GraphExtractor::loadImage(cv::Mat &image) {
  cv::Mat resized;
  cv::Size dsize(200, 100);

  cv::resize(image, resized, dsize, 0, 0, cv::INTER_LINEAR);

  cv::Mat gray, binary;
  cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);

  // cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
  //                       cv::THRESH_BINARY_INV, 11, 2);

  cv::threshold(gray, binary, 60, 255, cv::THRESH_BINARY_INV);

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

std::vector<cv::Point> GraphExtractor::extractGreen(cv::Mat &image) {
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

  cv::Scalar lower_green(35, 40, 40);
  cv::Scalar upper_green(85, 255, 255);

  cv::Mat mask;
  cv::inRange(hsv, lower_green, upper_green, mask);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Point> centers;

  for (const auto &contour : contours) {
    double area = cv::contourArea(contour);

    if (area <= 100)
      continue;

    cv::Moments m = cv::moments(contour);

    if (m.m00 == 0)
      continue;

    int cX = static_cast<int>(m.m10 / m.m00);
    int cY = static_cast<int>(m.m01 / m.m00);

    centers.push_back(cv::Point(cX, cY));
  }

  return centers;
}

cv::Point GraphExtractor::cvtPoint(cv::Mat &src, cv::Mat &dst,
                                   cv::Point point) {
  double sx = static_cast<double>(dst.cols) / static_cast<double>(src.cols);
  double sy = static_cast<double>(dst.rows) / static_cast<double>(src.rows);

  return cv::Point(cv::saturate_cast<int>(point.x * sx),
                   cv::saturate_cast<int>(point.y * sy));
}

Node *Graph::nodeFromID(int id) {
  for (Node &node : this->nodes) {
    if (node.id == id)
      return &node;
  }

  return nullptr;
}

TrackedNode *TrackedGraph::nodeFromID(int id) {
  for (TrackedNode &node : this->nodes) {
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
  this->removeUnconnectedNodes();
  this->updateGraph();
}

std::vector<Node> GraphExtractor::getNodes() { return this->graph.nodes; }

std::vector<Edge> GraphExtractor::getEdges() { return this->graph.edges; }

std::vector<TrackedNode> GraphExtractor::getTrackedNodes() {
  return this->trackedGraph.nodes;
}

std::vector<TrackedEdge> GraphExtractor::getTrackedEdges() {
  return this->trackedGraph.edges;
}

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
    node.id = this->graph.nextID++;

    if (surroundingPoints.size() > 3) {
      node.is_endpoint = false;
    } else {
      node.is_endpoint = true;
    }

    if (node.pos.x <= 1 || node.pos.y <= 1 || node.pos.x >= image.cols - 2 ||
        node.pos.y >= image.rows - 2) {
      node.screen_edge = true;
    } else {
      node.screen_edge = false;
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

std::vector<Edge *> Graph::getConnectedEdges(int nodeID) {
  std::vector<Edge *> result;

  for (Edge &edge : this->edges) {
    if (edge.src == nodeID || edge.dst == nodeID)
      result.push_back(&edge);
  }

  return result;
}

std::vector<TrackedEdge *> TrackedGraph::getConnectedEdges(int nodeID) {
  std::vector<TrackedEdge *> result;

  for (TrackedEdge &edge : this->edges) {
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

    Node *src = this->graph.nodeFromID(edges[i].src);
    Node *dst = this->graph.nodeFromID(edges[i].dst);

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
    for (Edge *connectedEdge : this->graph.getConnectedEdges(edges[i].src)) {
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

void GraphExtractor::removeUnconnectedNodes() {
  for (int i = 0; i < this->graph.nodes.size(); i++) {
    Node node = this->graph.nodes[i];

    bool connected = false;
    for (int j = 0; j < this->graph.edges.size() && !connected; j++) {
      Edge edge = this->graph.edges[j];

      if (edge.src == node.id || edge.dst == node.id)
        connected = true;
    }

    if (!connected) {
      this->graph.nodes.erase(this->graph.nodes.begin() + i);
      i--;
    }
  }
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
  Node current = path[path.size() - 1];
  Node previous = path[path.size() - 2];

  std::vector<Edge *> connected = this->graph.getConnectedEdges(current.id);
  std::vector<int> connectedNodes;

  for (const Edge *edge : connected) {
    if (edge->dst == current.id)
      connectedNodes.push_back(edge->src);
    else
      connectedNodes.push_back(edge->dst);
  }

  if (connected.size() == 0 || path.size() > this->pathLimit) {
    return;
  }

  std::vector<double> connectedDirs =
      this->getEdgeDirections(current, connected);

  double previousAngle = 0;

  for (int i = 0; i < connected.size(); i++) {
    if (connected[i]->src == previous.id || connected[i]->dst == previous.id) {
      previousAngle = connectedDirs[i];
    }
  }

  double targetAngle = fmod(previousAngle + M_PI, 2 * M_PI);
  double closestAngle = connectedDirs[0];
  int closestNode = connectedNodes[0];

  for (int i = 0; i < connected.size(); i++) {
    double angle = connectedDirs[i];
    if (abs(angle - targetAngle) < abs(closestAngle - targetAngle)) {
      closestAngle = angle;
      closestNode = connectedNodes[i];
    }
  }

  Node next = *this->graph.nodeFromID(closestNode);
  path.push_back(next);
  this->findNextNode(path);
}

void GraphExtractor::updateGraph() {
  for (TrackedNode &node : this->trackedGraph.nodes) {
    cv::Mat prediction = node.kf.predict();

    int x = prediction.at<float>(0);
    int y = prediction.at<float>(1);

    // node.pos = cv::Point(x, y);
  }

  if (this->trackedGraph.nodes.size() == 0) {
    graph.nextID = 0;

    for (const Node &node : this->graph.nodes) {
      TrackedNode newNode(node.pos);

      newNode.id = graph.nextID++;
      newNode.is_endpoint = node.is_endpoint;
      newNode.screen_edge = node.screen_edge;

      this->trackedGraph.nodes.push_back(newNode);
    }

    // TODO add edges
    return;
  }

  std::vector<std::vector<double>> costMatrix =
      this->trackedGraph.getCostMatrix(this->graph);
  std::vector<int> assignment;

  std::cout << costMatrix.size() << ", " << costMatrix[0].size() << std::endl;
  HungarianAlgorithm().Solve(costMatrix, assignment);
  std::vector<bool> matched(this->graph.nodes.size(), false);

  for (int i = 0; i < this->trackedGraph.nodes.size(); i++) {
    int assigned = assignment[i];

    if (assigned != -1 && costMatrix[i][assigned] <= this->gatingThreshold) {
      cv::Mat measurement =
          (cv::Mat_<float>(2, 1) << this->graph.nodes[assigned].pos.x,
           this->graph.nodes[assigned].pos.y);

      this->trackedGraph.nodes[i].kf.correct(measurement);
      this->trackedGraph.nodes[i].missedFrames = 0;
      this->trackedGraph.nodes[i].age++;
      this->trackedGraph.nodes[i].pos = this->graph.nodes[assigned].pos;
      matched[assigned] = true;
    }

    else {
      this->trackedGraph.nodes[i].missedFrames++;
    }
  }

  for (int i = 0; i < matched.size(); i++) {
    if (matched[i])
      continue;

    Node detectedNode = this->graph.nodes[assignment[i]];

    TrackedNode newNode(detectedNode.pos);
    newNode.id = this->trackedGraph.nextID++;
    newNode.screen_edge = detectedNode.screen_edge;
    newNode.is_endpoint = detectedNode.is_endpoint;

    this->trackedGraph.nodes.push_back(newNode);
  }

  this->trackedGraph.nodes.erase(
      std::remove_if(
          this->trackedGraph.nodes.begin(), this->trackedGraph.nodes.end(),
          [](const TrackedNode &node) { return node.missedFrames > 5; }),
      this->trackedGraph.nodes.end());
}

std::vector<double>
GraphExtractor::getEdgeDirections(Node origin, std::vector<Edge *> edges) {
  std::vector<double> results;

  for (const Edge *edge : edges) {
    cv::Point p;

    if (edge->src == origin.id) {
      p = edge->path[this->minEdgeSize - 1];
    }

    else {
      p = edge->path[edge->path.size() - this->minEdgeSize];
    }

    double dy = p.y - origin.pos.y;
    double dx = p.x - origin.pos.x;

    double angle = std::atan2(dy, dx);

    results.push_back(angle);
  }

  return results;
}

std::vector<Node> GraphExtractor::findPath(Node startPos) {
  std::vector<Node> path;

  // TODO find nearest node instead of only using exact position
  std::vector<Edge *> connectedEdges =
      this->graph.getConnectedEdges(startPos.id);

  if (connectedEdges.size() > 0) {
    Node next;

    if (connectedEdges[0]->src == startPos.id) {
      next = *this->graph.nodeFromID(connectedEdges[0]->src);
    }

    else {
      next = *this->graph.nodeFromID(connectedEdges[0]->dst);
    }

    path.push_back(startPos);
    path.push_back(next);

    this->findNextNode(path);
  }

  return path;
}
