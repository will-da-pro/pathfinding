#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

struct Node {
  int id;
  cv::Point pos; // averaged position after merging
  bool is_endpoint;
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

struct Graph {
  std::vector<Node> nodes;
  std::vector<Edge> edges;
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
  void loadImage(cv::Mat image);
  void processImage();
  std::vector<Node> getNodes();
  std::vector<Edge> getEdges();
  cv::Mat getSkeletonizedImage();
  std::vector<cv::Point> findPath(Node startPos);

  int pathLimit;

private:
  void extractNodes();
  void extractEdges();
  std::vector<cv::Point> getSurroundingPoints(cv::Point centre, int radius);
  std::vector<Edge> getConnectedEdges(Node node);
  Node followToNode(std::vector<cv::Point> &path);
  void findNextNode(std::vector<cv::Point> &path);
  double calculateAngle(cv::Point point1, cv::Point point2);

  cv::Mat rawImage;
  cv::Mat skeletonizedImage;

  Graph graph;
};
