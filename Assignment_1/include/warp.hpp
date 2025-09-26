#pragma once
#include <opencv2/core.hpp>
namespace vc {
cv::Mat warpPerspectiveCustom(const cv::Mat& src, const cv::Mat& H, cv::Size outSize);
}
