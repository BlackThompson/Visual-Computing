#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "blend.hpp"
namespace vc {
enum class Detector { SIFT, ORB, AKAZE };
cv::Mat stitchImages(const std::vector<cv::Mat>& imgs,
                     Detector detector,
                     vc::BlendMode blendMode,
                     int ransacIter,
                     double reprojThresh,
                     double ratio,
                     bool debug,
                     const std::string& outDir,
                     const std::string& setId,
                     const std::string& pairId);
}
