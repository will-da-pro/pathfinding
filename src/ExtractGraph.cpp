#include "pathfinding/ExtractGraph.hpp"
#include "Hungarian.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <numbers>
#include <opencv2/geometry.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdexcept>
#include <vector>

TrackedNode::TrackedNode(cv::Point pos) {
  // Initialize Kalman filter
  float dt = 1.0f;
  kf.transitionMatrix = cv::Mat_<float>(
      {4, 4}, {1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1});

  kf.measurementMatrix = cv::Mat_<float>({2, 4}, {1, 0, 0, 0, 0, 1, 0, 0});

  cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));

  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));

  cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

  kf.statePost.at<float>(1) = pos.x; // Initial X
  kf.statePost.at<float>(1) = pos.y; // Initial Y
  kf.statePost.at<float>(2) = 0.0f;  // Initial velocity X
  kf.statePost.at<float>(3) = 0.0f;  // Initial velocity Y

  // Initialize position
  this->pos = pos;
}

std::vector<std::vector<double>> TrackedGraph::getCostMatrix(Graph &graph) {
  /**
   * @brief Get the cost matrix of a graph against the tracked graph.
   */
  // The vector must be square
  // Choose size based on longest edge
  int size = this->nodes.size() > graph.nodes.size() ? this->nodes.size()
                                                     : graph.nodes.size();

  // Initialize square vector with correct size
  std::vector<std::vector<double>> costs(size, std::vector(size, 0.0));

  // Populate the cost matrix, where costs[i][j] is the cost between
  // graph.nodes[i] and trackedGraph.nodes[j]
  for (int i = 0; i < this->nodes.size(); i++) {
    for (int j = 0; j < graph.nodes.size(); j++) {
      Node &newNode = graph.nodes[j];
      Node &trackedNode = this->nodes[i];

      // Set the cost as the euclidean distance between points
      double distance = cv::norm(trackedNode.pos - newNode.pos);

      // Apply a penalty for the number of connected edges to improve accuracy
      int connectedEdgeDiff = graph.getConnectedEdges(newNode.id).size() -
                              this->getConnectedEdges(trackedNode.id).size();
      double penalty = trackedNode.screen_edge
                           ? 0
                           : this->edgePenalty * std::abs(connectedEdgeDiff);

      // Calculate final cost with penalty
      double cost = distance + penalty;
      costs[i][j] = cost;
    }
  }

  return costs;
}

GraphExtractor::GraphExtractor() {
  std::cout << "Initialising graph extractor..." << std::endl;
  this->pathLimit = 5;
  this->minEdgeSize = 25;
  this->gatingThreshold = 50;

  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  this->threshWriter =
      cv::VideoWriter("thresh.mp4", fourcc, 30, cv::Size(200, 100));

  this->pathWriter =
      cv::VideoWriter("path.mp4", fourcc, 30, cv::Size(200, 100));
}

void GraphExtractor::loadImage(cv::Mat &image) {
  /**
   * @brief Load the frame into GraphExtractor and apply skeletonization.
   *
   * Reduces size to 100x200px for performance, applies an adaptive threshold,
   * opens and closes the image to remove noise, and applies skeletonization.
   *
   * @param image Input frame.
   */
  cv::Mat resized;
  cv::Size dsize(200, 100);

  cv::resize(image, resized, dsize, 0, 0, cv::INTER_LINEAR);

  // Convert to gray and blur
  cv::Mat gray, binary;
  cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  GaussianBlur(gray, gray, cv::Size(5, 5), 0);

  // cv::Mat flat;
  // flat = this->flattenIllumination(gray);

  // Apply adaptive threshold
  cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY_INV, 55, 10);

  // cv::imshow("gray", flat);

  // cv::threshold(gray, binary, 120, 255, cv::THRESH_BINARY_INV);
  // binary = this->applySmoothVariableThreshold(gray);

  // cv::imshow("binary", binary);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

  // Remove noise from image
  cv::Mat opened_image;
  cv::morphologyEx(binary, opened_image, cv::MORPH_OPEN, kernel);

  cv::Mat closed_image;
  cv::morphologyEx(opened_image, closed_image, cv::MORPH_CLOSE, kernel);

  // Apply skeletonization
  cv::Mat skeleton;
  cv::ximgproc::thinning(closed_image, skeleton,
                         cv::ximgproc::THINNING_GUOHALL);

  int rows = skeleton.rows;
  int cols = skeleton.cols;

  // Set borders to 0
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

cv::Mat GraphExtractor::flattenIllumination(const cv::Mat &grayImage) {
  cv::Mat localBackground;

  // 1. Create a massive kernel (larger than the width of your black line)
  // This will completely "swallow" the black line, leaving ONLY the floor and
  // the glare.
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(41, 41));
  cv::morphologyEx(grayImage, localBackground, cv::MORPH_CLOSE, kernel);

  // cv::imshow("bg", localBackground);

  // At this point, 'localBackground' is an image of your floor
  // AS IF THE BLACK LINE WASN'T EVEN THERE. It contains just the glare and dark
  // corners.

  // 2. Divide the original image by the background illumination map
  // Formula: Corrected = (Original / Background) * 255
  cv::Mat corrected;
  cv::divide(grayImage, localBackground, corrected, 255.0);

  return corrected;
}

cv::Mat GraphExtractor::applySmoothVariableThreshold(const cv::Mat &grayImage) {
  int rows = grayImage.rows;
  int cols = grayImage.cols;

  // 1. Create a threshold map matrix (CV_8UC1) filled with default values
  cv::Mat thresholdMap = cv::Mat::zeros(grayImage.size(), CV_8UC1);

  cv::Point center(cols / 2, rows / 2);
  double maxDist = std::sqrt(center.x * center.x + center.y * center.y);

  int centerThresh = 160; // High threshold in the middle
  int edgeThresh = 70;    // Low threshold at the corners

  // 2. Populate the threshold map based on distance from the center
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      double dist = std::sqrt((x - center.x) * (x - center.x) +
                              (y - center.y) * (y - center.y));
      double factor =
          std::pow(dist / maxDist, 2); // 0.0 at center, 1.0 at furthest corner

      // Linear interpolation between center and edge thresholds
      int customThresh = centerThresh - (factor * (centerThresh - edgeThresh));
      thresholdMap.at<uchar>(y, x) = static_cast<uchar>(customThresh);
    }
  }

  // 3. Compare the image directly with the custom threshold map
  // If grayImage(y,x) > thresholdMap(y,x), result is 255, else 0
  cv::Mat binaryOutput;
  cv::compare(grayImage, thresholdMap, binaryOutput, cv::CMP_LT);

  return binaryOutput;
}

std::vector<cv::Point> GraphExtractor::extractGreen(cv::Mat &image) {
  /**
   * @brief Extract all green areas from the image.
   *
   * Applies a HSV threshold and contours to detect all green contours of a
   * great enough area.
   *
   * @param image Input frame.
   * @return Vector of all green centeroids.
   */
  // Convert to HSV
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

  // Green range
  cv::Scalar lower_green(35, 40, 40);
  cv::Scalar upper_green(85, 255, 255);

  // Apply green mask
  cv::Mat mask;
  cv::inRange(hsv, lower_green, upper_green, mask);

  // Reduce noise
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

  // Find green contours
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Point> centers;

  // Get centers of all green contours and check if large enough
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
  /**
   * @brief Convert a point from the same relative position in src to dst.
   */
  double sx = static_cast<double>(dst.cols) / static_cast<double>(src.cols);
  double sy = static_cast<double>(dst.rows) / static_cast<double>(src.rows);

  return cv::Point(cv::saturate_cast<int>(point.x * sx),
                   cv::saturate_cast<int>(point.y * sy));
}

Node *Graph::nodeFromID(int id) {
  /**
   * @brief Get a node object from its ID.
   */
  for (Node &node : this->nodes) {
    if (node.id == id)
      return &node;
  }

  return nullptr;
}

TrackedNode *TrackedGraph::nodeFromID(int id) {
  /**
   * @brief Get a tracked node object from its ID.
   */
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
  this->updateGreen();
  this->findNextTarget(this->currentTarget, &this->currentEdge);
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

    if (src->screen_edge || dst->screen_edge)
      continue;

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
  /**
   * @brief Merge 2 connected edges into one.
   *
   * Connects both path variables, and updates src, dst and length. Edges must
   * be connected, i.e. they must share a common endpoint (src/dst).
   *
   * @param edge1 First edge.
   * @param edge2 Second edge.
   * @return Combined edge.
   */
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
  /**
   * @brief Remove all nodes not connected to an edge.
   */
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
  /**
   * @brief Find all edges connected to a node.
   *
   * Traces over the skeletonized image to find connected nodes, and turns the
   * traces into edges.
   *
   * @param node Node to trace edges from.
   * @return Connected edges.
   */
  // Get all points surrounding current point
  std::vector<Edge> connectedEdges;
  std::vector<cv::Point> surroundingPoints =
      this->getSurroundingPoints(node.pos, 3);

  // Iterate over surrounding points
  for (const auto &point : surroundingPoints) {
    // Ignore the current point
    if (point == node.pos)
      continue;

    Edge edge;
    edge.src = node.id;

    edge.path.push_back(node.pos);
    edge.path.push_back(point);

    // Follow path until the next node
    edge.dst = this->followToNode(edge.path).id;
    edge.length = edge.path.size();

    connectedEdges.push_back(edge);
  }

  return connectedEdges;
}

double GraphExtractor::calculateAngle(cv::Point point1, cv::Point point2) {
  /**
   * @brief Calculate the angle of point2 from point1.
   *
   * Calculates the angle of point2 relative to point1. Returns 0 if point1 ==
   * point2.
   *
   * @param point1 First point.
   * @param point2 Second point.
   * @return Angle of second point from first point.
   */
  int rise = point2.y - point1.y;
  int run = point2.x - point1.x;

  double angle = std::atan2(rise, run);
  return angle;
}

double GraphExtractor::calculateDist(cv::Point point1, cv::Point point2) {
  return std::sqrt(std::pow(point1.x - point2.x, 2) +
                   std::pow(point1.y - point2.y, 2));
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
  /**
   * @brief Update the tracked graph with the observed graph.
   *
   * Applies the Hungarian algorithm to the observed and tracked nodes
   * to find the lowest cost pairings of nodes between frames to track
   * them.
   *
   */
  // Update Kalman filter for all tracked nodes
  for (TrackedNode &node : this->trackedGraph.nodes) {
    cv::Mat prediction = node.kf.predict();

    int x = prediction.at<float>(0);
    int y = prediction.at<float>(1);

    // node.pos = cv::Point(x, y);
  }

  // If there are no nodes, add all nodes currently observed
  if (this->trackedGraph.nodes.size() == 0) {
    graph.nextID = 0;

    std::vector<int> newIDs;

    // Add every observed node to the tracked graph
    for (const Node &node : this->graph.nodes) {
      TrackedNode newNode(node.pos);

      // Give the node a new ID
      newNode.id = graph.nextID++;
      newNode.is_endpoint = node.is_endpoint;
      newNode.screen_edge = node.screen_edge;

      // Add to tracked graph
      this->trackedGraph.nodes.push_back(newNode);

      // Add the new ID to a vector so it can be used to update tracked edges
      newIDs.push_back(newNode.id);
    }

    // Iterate over every known edge
    for (int i = 0; i < this->graph.edges.size(); i++) {
      Edge edge = this->graph.edges[i];
      TrackedEdge trackedEdge;
      this->edgeToTracked(edge, trackedEdge);

      // Get the IDs of the edge dst and src in the observed graph
      int src = edge.src;
      int dst = edge.dst;

      // Find the index of the dst and src nodes within the observed graph
      auto src_it =
          std::find_if(this->graph.nodes.begin(), this->graph.nodes.end(),
                       [&src](const Node &node) { return node.id == src; });

      auto dst_it =
          std::find_if(this->graph.nodes.begin(), this->graph.nodes.end(),
                       [&dst](const Node &node) { return node.id == dst; });

      // Handle any edge case where the nodes don't exist
      if (src_it == this->graph.nodes.end() ||
          dst_it == this->graph.nodes.end()) {
        continue;
      }

      // Find the new IDs of the nodes in the tracked graph
      int srcIndex = newIDs[std::distance(this->graph.nodes.begin(), src_it)];
      int dstIndex = newIDs[std::distance(this->graph.nodes.begin(), dst_it)];

      // Update the new tracked edge and add to the tracked graph
      trackedEdge.src = newIDs[srcIndex];
      trackedEdge.dst = newIDs[dstIndex];

      this->trackedGraph.edges.push_back(trackedEdge);
    }

    return;
  }

  // Match observed nodes to tracked nodes
  std::vector<std::vector<double>> costMatrix =
      this->trackedGraph.getCostMatrix(this->graph);
  std::vector<int> assignment;

  // Apply Hungarian algorithm to find lowest cost pairings of nodes between
  // frames
  HungarianAlgorithm().Solve(costMatrix, assignment);
  std::vector<bool> matched(this->graph.nodes.size(), false);
  std::vector<int> newIDs(this->graph.nodes.size(), 0);

  // Update tracked graph with new matches
  for (int i = 0; i < this->trackedGraph.nodes.size(); i++) {
    int assigned = assignment[i];

    // Check if a tracked node was observed in this frame
    if (assigned != -1 && assigned < this->graph.nodes.size() &&
        costMatrix[i][assigned] <= this->gatingThreshold) {

      // cv::Mat measurement = cv::Mat_<float>(
      //     {2, 1}, {static_cast<float>(this->graph.nodes[assigned].pos.x),
      //              static_cast<float>(this->graph.nodes[assigned].pos.y)});
      // this->trackedGraph.nodes[i].kf.correct(measurement);

      // Update the tracked node with data from the observed node
      this->trackedGraph.nodes[i].missedFrames = 0;
      this->trackedGraph.nodes[i].age++;
      this->trackedGraph.nodes[i].pos = this->graph.nodes[assigned].pos;
      this->trackedGraph.nodes[i].is_endpoint =
          this->graph.nodes[assigned].is_endpoint;
      this->trackedGraph.nodes[i].screen_edge =
          this->graph.nodes[assigned].screen_edge;
      matched[assigned] = true;
      newIDs[assigned] = this->trackedGraph.nodes[i].id;
    }

    // If it was not observed, update its missed frames
    else {
      this->trackedGraph.nodes[i].missedFrames++;
    }
  }

  // If there was no valid match found for a node, add it to the tracked graph
  for (int i = 0; i < matched.size(); i++) {
    if (matched[i])
      continue;

    // Get the new observed node
    Node detectedNode = this->graph.nodes[i];

    // Create a new tracked node
    TrackedNode newNode(detectedNode.pos);
    newNode.id = this->trackedGraph.nextID++;
    newNode.screen_edge = detectedNode.screen_edge;
    newNode.is_endpoint = detectedNode.is_endpoint;

    // Add the node to the tracked graph
    this->trackedGraph.nodes.push_back(newNode);
    newIDs[i] = newNode.id;
  }

  // Remove nodes that haven't been seen in 5 frames, and nodes that are less
  // than 5 frames old that haven't been seen in 1 or more frames
  this->trackedGraph.nodes.erase(
      std::remove_if(this->trackedGraph.nodes.begin(),
                     this->trackedGraph.nodes.end(),
                     [](const TrackedNode &node) {
                       return node.missedFrames > 5 ||
                              (node.age < 5 && node.missedFrames > 0);
                     }),
      this->trackedGraph.nodes.end());

  // Remove all tracked edges
  this->trackedGraph.edges.clear();

  // Add edges to tracked graph
  // Iterate over every node
  for (int i = 0; i < this->graph.nodes.size(); i++) {
    Node &node = this->graph.nodes[i];

    // Iterate over every edge
    for (const Edge &edge : this->graph.edges) {
      // If edge is not connected to the current node, continue
      if (edge.src != node.id && edge.dst != node.id) {
        continue;
      }

      // Get the ID of the other connected node
      int connectedID = edge.src == node.id ? edge.dst : edge.src;

      // Make sure the other node exists and make an iterator to it
      auto connectedIt =
          std::find_if(this->graph.nodes.begin(), this->graph.nodes.end(),
                       [&connectedID](const Node &connected) {
                         return connected.id == connectedID;
                       });

      if (connectedIt == this->graph.nodes.end()) {
        std::cerr
            << "Couldn't find the other node??? (This should never happen)\n";
        return;
      }

      // Get the index of the connected node within the observed graph
      int connectedIndex =
          std::distance(this->graph.nodes.begin(), connectedIt);

      // Get the new tracked node IDs within the tracked graph
      int trackedSrcIndex = edge.src == node.id ? i : connectedIndex;
      int trackedDstIndex = edge.dst == node.id ? i : connectedIndex;

      int trackedSrc = newIDs[trackedSrcIndex];
      int trackedDst = newIDs[trackedDstIndex];

      // Create a new tracked edge
      TrackedEdge tracked;
      this->edgeToTracked(edge, tracked);

      tracked.src = trackedSrc;
      tracked.dst = trackedDst;

      // Make sure the edge doesn't already exist to prevent duplicates
      bool exists = false;
      for (const TrackedEdge &existingTracked : this->trackedGraph.edges) {
        if ((tracked.src == existingTracked.src &&
             tracked.dst == existingTracked.dst) ||
            (tracked.src == existingTracked.dst &&
             tracked.dst == existingTracked.src)) {
          exists = true;
          break;
        }
      }

      if (exists) {
        continue;
      }

      // Add the new tracked edge to the tracked graph
      this->trackedGraph.edges.push_back(tracked);
    }
  }

  // Remove unconnected edges
  for (int i = 0; i < this->trackedGraph.edges.size(); i++) {
    const TrackedEdge &edge = this->trackedGraph.edges[i];

    int src = edge.src;
    int dst = edge.dst;

    auto src_it = std::find_if(
        this->trackedGraph.nodes.begin(), this->trackedGraph.nodes.end(),
        [&src](const TrackedNode &node) { return node.id == src; });

    auto dst_it = std::find_if(
        this->trackedGraph.nodes.begin(), this->trackedGraph.nodes.end(),
        [&dst](const TrackedNode &node) { return node.id == dst; });

    if (src_it != this->trackedGraph.nodes.end() &&
        dst_it != this->trackedGraph.nodes.end()) {
      continue;
    }

    this->trackedGraph.edges.erase(this->trackedGraph.edges.begin() + i);
    i--;
  }
}

void GraphExtractor::updateGreen() {
  std::vector<cv::Point> greenCenters = this->extractGreen(this->rawImage);

  for (cv::Point &green : greenCenters) {
    green = this->cvtPoint(this->rawImage, this->skeletonizedImage, green);
  }

  this->green = greenCenters;
}

std::vector<double>
GraphExtractor::getEdgeDirections(Node origin, std::vector<Edge *> edges) {
  /**
   * @brief Get the direction of edges relative to a central node.
   *
   * Iterates over each edge, and finds the angle of the edge relative to the
   * origin. Assumes that all edges are connected to the origin.
   *
   * @param origin Central node to get directions from.
   * @param[in] edges Edges to get the directions of.
   * @return A vector the same size as edges where output[i] is the angle of
   * edges[i] from origin.
   */
  std::vector<double> results;

  for (const Edge *edge : edges) {
    cv::Point p;

    Node *src = this->graph.nodeFromID(edge->src);
    Node *dst = this->graph.nodeFromID(edge->dst);

    if (!src || !dst) {
      throw runtime_error("src or dst didn't exist on edge");
    }

    bool screenEdge = src->screen_edge || dst->screen_edge;

    if (edge->src == origin.id) {
      p = screenEdge ? dst->pos : edge->path[this->minEdgeSize - 1];
    }

    else {
      p = screenEdge ? src->pos
                     : edge->path[edge->path.size() - this->minEdgeSize];
    }

    double dy = p.y - origin.pos.y;
    double dx = p.x - origin.pos.x;

    double angle = std::atan2(dy, dx);

    results.push_back(angle);
  }

  return results;
}

void GraphExtractor::edgeToTracked(const Edge &edge, TrackedEdge &tracked) {
  /**
   * @brief Convert an observed edge into a tracked edge.
   *
   * Copies length, angles from src and dst, and path.
   *
   * @param[in] edge Edge to convert.
   * @param[out] tracked Tracked edge to output edge data to.
   */
  tracked.length = edge.length;
  tracked.age = 0;

  Node *src = this->graph.nodeFromID(edge.src);
  Node *dst = this->graph.nodeFromID(edge.dst);

  if (!src || !dst) {
    std::cerr << "Src or dst doesn't exist on edge\n";
    return;
  }

  if (src->screen_edge || dst->screen_edge) {
    tracked.angleFromSrc = this->calculateAngle(src->pos, dst->pos);
    tracked.angleFromDst = this->calculateAngle(dst->pos, src->pos);
  } else {
    tracked.angleFromSrc =
        this->calculateAngle(src->pos, edge.path[this->minEdgeSize - 1]);
    tracked.angleFromDst = this->calculateAngle(
        dst->pos, edge.path[edge.path.size() - this->minEdgeSize]);
  }

  tracked.path = edge.path;
}

std::vector<Node> GraphExtractor::findPath(Node startPos) {
  /**
   * @brief find path using only the highest node in the frame.
   *
   * Old path finding algorithm, should not be used.
   *
   * @param startPos Node to start on.
   */
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

void GraphExtractor::findStartingEdge(
    int &trackingID, std::shared_ptr<TrackedEdge> *currentEdge) {
  /**
   * @brief Find a new starting edge for pathfinging.
   *
   * Finds the longest edge and follows in the direction up the screen.
   *
   * @param[out] trackingID ID of the node to track.
   * @param[out] currentEdge Current edge to track.
   */
  trackingID = -1;

  if (this->trackedGraph.edges.size() == 0)
    return;

  int largest = 0;
  for (TrackedEdge &edge : this->trackedGraph.edges) {
    if (edge.length > largest) {
      *currentEdge = std::make_shared<TrackedEdge>(edge);
      largest = edge.length;
    }
  }

  TrackedNode *src = this->trackedGraph.nodeFromID((*currentEdge)->src);
  TrackedNode *dst = this->trackedGraph.nodeFromID((*currentEdge)->dst);

  trackingID =
      src->pos.y > dst->pos.y ? (*currentEdge)->dst : (*currentEdge)->src;
}

double GraphExtractor::wrapAngle(double angle) {
  /**
   * @brief Wrap angle within the range (-pi, pi].
   *
   * @param angle Angle to wrap.
   * @return Wrapped angle.
   */
  constexpr double two_pi = 2.0 * std::numbers::pi;
  double wrapped = std::remainder(angle, two_pi);

  if (wrapped == -std::numbers::pi)
    wrapped = std::numbers::pi;

  return wrapped;
}

double GraphExtractor::addAngles(double angle1, double angle2) {
  /**
   * @brief Adds 2 angles and wraps them.
   *
   * @param angle1 First angle.
   * @param angle2 Second angle.
   * @return Wrapped sum of angles.
   */
  double sum = angle1 + angle2;
  double wrapped = this->wrapAngle(sum);
  return wrapped;
}

void GraphExtractor::findNextTarget(int &trackingID,
                                    std::shared_ptr<TrackedEdge> *currentEdge) {
  /**
   * @brief Find the next node to track based on the tracked graph.
   *
   * @param[in,out] trackingID Current node being tracked.
   * @param[in,out] currentEdge Current edge being tracked.
   */
  // Handle breaks in the line.
  if (this->searchLineBreak) {
    std::cout << "Line break\n";
    // Return if there are no nodes being tracked currently.
    if (this->trackedGraph.nodes.size() == 0)
      return;

    std::vector<TrackedNode> nodesInRange;

    // Find all nodes within range of the projected line.
    for (TrackedNode &node : this->trackedGraph.nodes) {
      double dist = this->searchDistance(node.pos);
      if (dist > this->searchMinDist)
        continue;

      nodesInRange.push_back(node);
    }

    // Return if no nodes are in range.
    if (nodesInRange.size() == 0)
      return;

    // Get the closest node to the projected line.
    TrackedNode closestNode = nodesInRange[0];
    double closestDist =
        this->calculateDist(this->searchLastPoint, closestNode.pos);

    for (TrackedNode &node : nodesInRange) {
      double dist = this->calculateDist(this->searchLastPoint, node.pos);

      if (dist > closestDist)
        continue;

      closestDist = dist;
      closestNode = node;
    }

    // Get edges surrounding closest node
    std::vector<TrackedEdge *> surroundingEdges;

    for (TrackedEdge &edge : this->trackedGraph.edges) {
      if (edge.src == closestNode.id || edge.dst == closestNode.id)
        surroundingEdges.push_back(&edge);
    }

    if (surroundingEdges.size() == 0) {
      return;
    }

    // Update edge and target
    std::shared_ptr<TrackedEdge> newEdge =
        std::make_shared<TrackedEdge>(*this->closestToAngle(
            closestNode.id, surroundingEdges, this->searchDirection));
    int newTarget =
        closestNode.id == newEdge->src ? newEdge->dst : newEdge->src;

    this->currentTarget = newTarget;
    this->currentEdge = newEdge;
    this->searchLineBreak = false;
  }

  // Reset tracked node and edge
  if (!currentEdge || trackingID < 0) {
    *currentEdge = std::make_shared<TrackedEdge>(TrackedEdge());
    this->findStartingEdge(trackingID, currentEdge);
    return;
  }

  TrackedNode *currentNodePointer = this->trackedGraph.nodeFromID(trackingID);

  // Make sure the current node exists
  if (!currentNodePointer) {

    this->currentTarget = -1;
    this->findStartingEdge(trackingID, currentEdge);
    return;
  }

  TrackedNode currentNode = *currentNodePointer;

  // Don't attempt to find another node if the node is on the screen edge.
  if (currentNode.screen_edge)
    return;
  // Handle line breaks
  else if (currentNode.is_endpoint) {
    this->searchLineBreak = true;
    this->searchLastNode = currentNode.id;
    this->searchLastPoint = currentNode.pos;
    double currentDir = trackingID == (*currentEdge)->src
                            ? (*currentEdge)->angleFromSrc
                            : (*currentEdge)->angleFromDst;
    this->searchDirection = this->addAngles(currentDir, std::numbers::pi);

    return;
  }

  // Get all edges surrounding the tracked node
  std::vector<TrackedEdge *> surroundingEdges;

  for (TrackedEdge &edge : this->trackedGraph.edges) {
    if (edge.src == trackingID || edge.dst == trackingID)
      surroundingEdges.push_back(&edge);
  }

  if (surroundingEdges.size() == 0) {
    std::cerr << "No surrounding edges to node.\n";
    return;
  }
  std::weak_ptr<TrackedEdge> weakRef = *currentEdge;
  if (auto lockedPtr = weakRef.lock())
    ;
  else {
    return;
  }

  // Get the angle of the tracked edge from the tracked node.
  double currentAngle = trackingID == (*currentEdge)->src
                            ? (*currentEdge)->angleFromSrc
                            : (*currentEdge)->angleFromDst;

  // Only 2 surrounding edges, not an intersection.
  if (surroundingEdges.size() < 3) {
    double targetAngle = this->addAngles(currentAngle, std::numbers::pi);

    TrackedEdge *closestEdge =
        this->closestToAngle(trackingID, surroundingEdges, targetAngle);

    *this->currentEdge = *closestEdge;
    this->currentTarget =
        trackingID == closestEdge->src ? closestEdge->dst : closestEdge->src;

    return;
  }

  // Find adjacent green squares
  std::vector<double> surroundingGreen;
  double minGreenDist = 40;

  for (cv::Point greenPos : this->green) {
    double dist = this->calculateDist(greenPos, currentNode.pos);
    if (dist > minGreenDist)
      continue;

    double angle = this->calculateAngle(currentNode.pos, greenPos);
    surroundingGreen.push_back(angle);
  }

  // Target angles for going straight, and turning left and right.
  double targetLeft = this->addAngles(currentAngle, -std::numbers::pi / 2);
  double targetRight = this->addAngles(currentAngle, std::numbers::pi / 2);
  double targetStraight = this->addAngles(currentAngle, std::numbers::pi);

  // Find closest edges to target angles.
  TrackedEdge *leftEdge =
      this->closestToAngle(trackingID, surroundingEdges, targetLeft);
  TrackedEdge *rightEdge =
      this->closestToAngle(trackingID, surroundingEdges, targetRight);
  TrackedEdge *straightEdge =
      this->closestToAngle(trackingID, surroundingEdges, targetStraight);

  // Check if there is green to left or right of line.
  bool greenLeft = false;
  bool greenRight = false;

  for (double &greenAngle : surroundingGreen) {
    double diff = this->addAngles(currentAngle, -greenAngle);

    if (diff < 0 && diff > -std::numbers::pi / 2)
      greenLeft = true;
    if (diff > 0 && diff < std::numbers::pi / 2)
      greenRight = true;
  }

  // Handle green turn logic.
  if (greenRight && greenLeft) {
    trackingID = trackingID == (*currentEdge)->src ? (*currentEdge)->dst
                                                   : (*currentEdge)->src;
    return;
  }

  else if (greenRight) {
    **currentEdge = *rightEdge;
  }

  else if (greenLeft) {
    **currentEdge = *leftEdge;
  }

  else {
    **currentEdge = *straightEdge;
  }

  trackingID = trackingID == (*currentEdge)->src ? (*currentEdge)->dst
                                                 : (*currentEdge)->src;
}

double GraphExtractor::searchDistance(cv::Point point) {
  double sinTheta = std::sin(this->searchDirection);
  double cosTheta = std::cos(this->searchDirection);

  // Vector from startPoint to targetPoint
  double dx = point.x - this->searchLastPoint.x;
  double dy = point.y - this->searchLastPoint.y;

  // Project the target point onto the line's direction vector (Dot Product)
  double projection = dx * cosTheta + dy * sinTheta;

  if (projection < 0.0) {
    // The point is "behind" the starting point.
    // Return the straight-line Euclidean distance to the startPoint.
    return 50 * std::sqrt(dx * dx + dy * dy);
  }

  return std::abs(dx * sinTheta - dy * cosTheta);
}

TrackedEdge *
GraphExtractor::closestToAngle(int currentNode,
                               std::vector<TrackedEdge *> currentEdges,
                               double targetAngle) {
  if (currentEdges.size() < 1) {
    return nullptr;
  }

  TrackedEdge *closestEdge = currentEdges[0];
  double closestAngle = currentNode == currentEdge->src
                            ? closestEdge->angleFromSrc
                            : closestEdge->angleFromDst;

  for (TrackedEdge *edge : currentEdges) {
    double angle =
        currentNode == edge->src ? edge->angleFromSrc : edge->angleFromDst;

    double closestDiff = std::abs(this->addAngles(targetAngle, -closestAngle));

    double diff = std::abs(this->addAngles(targetAngle, -angle));

    if (diff < closestDiff) {
      closestAngle = angle;
      closestEdge = edge;
    }
  }

  return closestEdge;
}

TrackedNode *GraphExtractor::getTarget() {
  return this->trackedGraph.nodeFromID(this->currentTarget);
}
