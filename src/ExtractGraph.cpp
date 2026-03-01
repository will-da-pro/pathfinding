#include "thinning/ExtractGraph.hpp"
#include <iostream>
#include <opencv2/ximgproc.hpp>
#include <vector>

GraphExtractor::GraphExtractor() {
  std::cout << "Initialising graph extractor..." << std::endl;
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

std::vector<std::pair<cv::Point, cv::Point>> GraphExtractor::getLines() {
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
  return surroundingPoints;
}

void GraphExtractor::extractLines() {}
