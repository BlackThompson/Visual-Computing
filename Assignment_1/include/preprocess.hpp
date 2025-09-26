#pragma once
#include <opencv2/core.hpp>
namespace vc {
cv::Mat toGray(const cv::Mat& src);
cv::Mat normalizeImage(const cv::Mat& src);
cv::Mat gaussianBlur(const cv::Mat& src, int ksize, double sigma);
}
