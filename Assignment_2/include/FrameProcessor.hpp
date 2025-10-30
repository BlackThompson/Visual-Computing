// Declares CPU-side filter and transformation utilities built on top of OpenCV.

#pragma once

#include "Types.hpp"

#include <opencv2/core.hpp>

namespace FrameProcessor
{
    // Applies the selected filter on CPU and returns a new frame in BGR format.
    cv::Mat applyFilter(const cv::Mat& frameBgr,
                        FilterType filter,
                        const FilterParameters& params);

    // Applies the affine transform in-place to the provided frame.
    void applyTransform(cv::Mat& frameBgr,
                        const TransformParams& transform);

    // Helper for computing the affine transform matrix (2x3) for warpAffine.
    cv::Mat computeAffineMatrix(const cv::Size& frameSize,
                                const TransformParams& transform);
}
