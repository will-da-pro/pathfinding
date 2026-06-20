#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

struct Node {
  int id;
  cv::Point pos; // averaged position after merging
  bool is_endpoint;
  bool screen_edge;
};

struct Edge {
  int src, dst;                // node IDs
  std::vector<cv::Point> path; // pixel chain along the skeleton
  double length;               // Euclidean arc length

  bool operator==(const Edge &other) const {
    return (src == other.src && dst == other.dst) ||
           (src == other.dst && dst == other.src);
  }
};

class Graph {
public:
  std::vector<Node> nodes;
  std::vector<Edge> edges;

  int nextID = 0;

  Node *nodeFromID(int id);
  std::vector<Edge *> getConnectedEdges(int nodeID);
};

struct LocalEdge : Edge {};

struct TrackedNode : Node {
  TrackedNode(cv::Point pos);
  int age = 0;
  int missedFrames = 0;
  cv::KalmanFilter kf = cv::KalmanFilter(4, 2, 0);
};

struct TrackedEdge : Edge {
  uint32_t age = 0;
};

class TrackedGraph {
public:
  std::vector<TrackedNode> nodes;
  std::vector<TrackedEdge> edges;

  int nextID = 0;

  TrackedNode *nodeFromID(int id);
  std::vector<TrackedEdge *> getConnectedEdges(int nodeID);

  std::vector<std::vector<double>> getCostMatrix(Graph &graph);
};

struct ComparePoints {
  bool operator()(const Node &a, const Node &b) const {
    if (a.pos.y != b.pos.y) {
      return a.pos.y < b.pos.y;
    }
    return a.pos.x < b.pos.x;
  }
};

class GraphExtractor {
public:
  GraphExtractor();

  void loadImage(cv::Mat &image);
  void processImage();
  std::vector<cv::Point> extractGreen(cv::Mat &image);
  cv::Point cvtPoint(cv::Mat &src, cv::Mat &dst, cv::Point point);

  std::vector<Node> getNodes();
  std::vector<Edge> getEdges();
  std::vector<TrackedNode> getTrackedNodes();
  std::vector<TrackedEdge> getTrackedEdges();

  cv::Mat getSkeletonizedImage();
  std::vector<Node> findPath(Node startPos);

  int pathLimit;
  int minEdgeSize;
  int gatingThreshold;

private:
  void extractNodes();
  void extractEdges();

  std::vector<cv::Point> getSurroundingPoints(cv::Point centre, int radius);
  std::vector<Edge> traceConnectedEdges(Node node);
  Node followToNode(std::vector<cv::Point> &path);

  void removeShortEdges(std::vector<Edge> &edges);
  Edge mergeEdges(Edge edge1, Edge edge2);
  void removeUnconnectedNodes();

  void findNextNode(std::vector<Node> &path);
  double calculateAngle(cv::Point point1, cv::Point point2);

  void updateGraph();

  std::vector<double> getEdgeDirections(Node origin, std::vector<Edge *> edges);

  cv::Mat rawImage;
  cv::Mat skeletonizedImage;

  Graph graph;
  TrackedGraph trackedGraph;
};
