#pragma once
#include <opencv2/core.hpp>
namespace vc {
enum class BlendMode { OVERLAY, FEATHER };
cv::Mat blendOverlay(const cv::Mat& baseImg, const cv::Mat& topImg, const cv::Mat& mask);
cv::Mat blendFeather(const cv::Mat& baseImg, const cv::Mat& topImg, const cv::Mat& weightMask, double eps=1e-6);
}
