#include <memory>
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
  double angleFromSrc;
  double angleFromDst;
};

class TrackedGraph {
public:
  std::vector<TrackedNode> nodes;
  std::vector<TrackedEdge> edges;

  int nextID = 0;
  int edgePenalty = 20; // TODO Change later once edge detection exists

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
  TrackedNode *getTarget();
  TrackedEdge *getCurrentEdge();

  cv::Mat getSkeletonizedImage();
  [[deprecated]] std::vector<Node> findPath(Node startPos);

  int pathLimit;
  int minEdgeSize;
  int gatingThreshold;
  TrackedGraph trackedGraph;

  cv::VideoWriter threshWriter;
  cv::VideoWriter pathWriter;

private:
  void extractNodes();
  void extractEdges();

  cv::Mat applySmoothVariableThreshold(const cv::Mat &grayImage);
  cv::Mat flattenIllumination(const cv::Mat &grayImage);

  std::vector<cv::Point> getSurroundingPoints(cv::Point centre, int radius);
  std::vector<Edge> traceConnectedEdges(Node node);
  Node followToNode(std::vector<cv::Point> &path);

  void removeShortEdges(std::vector<Edge> &edges);
  Edge mergeEdges(Edge edge1, Edge edge2);
  void removeUnconnectedNodes();

  void findNextNode(std::vector<Node> &path);
  double calculateAngle(cv::Point point1, cv::Point point2);
  double calculateDist(cv::Point point1, cv::Point point2);

  void updateGraph();
  void updateGreen();

  std::vector<double> getEdgeDirections(Node origin, std::vector<Edge *> edges);

  void edgeToTracked(const Edge &edge, TrackedEdge &trackedEdge);

  double wrapAngle(double angle);
  double addAngles(double angle1, double angle2);

  void findStartingEdge(int &trackingID,
                        std::shared_ptr<TrackedEdge> *currentEdge);
  void findNextTarget(int &trackingID,
                      std::shared_ptr<TrackedEdge> *currentEdge);
  TrackedEdge *closestToAngle(int currentNode,
                              std::vector<TrackedEdge *> currentEdges,
                              double targetAngle);

  double searchDistance(cv::Point point);

  cv::Mat rawImage;
  cv::Mat skeletonizedImage;

  Graph graph;
  std::vector<cv::Point> green;

  int currentTarget = -1;
  std::shared_ptr<TrackedEdge> currentEdge = nullptr;

  bool searchLineBreak = false;
  int searchLastNode;
  cv::Point searchLastPoint;
  double searchDirection;
  double searchMinDist = 10;
};
