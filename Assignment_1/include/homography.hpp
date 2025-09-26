#pragma once
#include <opencv2/core.hpp>
#include <vector>
namespace vc {
cv::Mat computeHomographyDLT(const std::vector<cv::Point2f>& srcPts, const std::vector<cv::Point2f>& dstPts);
cv::Mat ransacHomography(const std::vector<cv::Point2f>& srcPts, const std::vector<cv::Point2f>& dstPts, int iterations, double thresh, std::vector<unsigned char>& inlierMask);
}
