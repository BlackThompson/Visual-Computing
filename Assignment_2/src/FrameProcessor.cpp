// Implements CPU-based filter and transformation routines using OpenCV.

#include "FrameProcessor.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace
{
    cv::Mat applyPixelate(const cv::Mat& frameBgr, int blockSize)
    {
        cv::Mat result;
        if (blockSize <= 1)
        {
            result = frameBgr.clone();
            return result;
        }

        cv::Mat downscaled;
        const double inv = 1.0 / static_cast<double>(blockSize);
        cv::resize(frameBgr, downscaled, cv::Size(), inv, inv, cv::INTER_LINEAR);
        cv::resize(downscaled, result, frameBgr.size(), 0.0, 0.0, cv::INTER_NEAREST);
        return result;
    }

    cv::Mat applyComic(const cv::Mat& frameBgr, const ComicParams& params)
    {
        cv::Mat blurred;
        cv::bilateralFilter(frameBgr, blurred, 9, 75, 75);

        cv::Mat gray;
        cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);

        // Edge detection using the Laplacian operator.
        cv::Mat edges;
        cv::Laplacian(gray, edges, CV_8U, 5);
        cv::Mat edgesInv;
        cv::threshold(edges, edgesInv,
                      params.edgeThreshold * 255.0f,
                      255.0,
                      cv::THRESH_BINARY_INV);

        // Posterise the colour palette.
        cv::Mat quantised = blurred.clone();
        const int levels = std::max(2, params.colorLevels);
        const int step = 255 / (levels - 1);
        quantised.forEach<cv::Vec3b>([step](cv::Vec3b& pixel, const int*)
        {
            for (int c = 0; c < 3; ++c)
            {
                pixel[c] = static_cast<uchar>(
                    cvRound(static_cast<float>(pixel[c]) / step) * step);
            }
        });

        cv::Mat edgesColor;
        cv::cvtColor(edgesInv, edgesColor, cv::COLOR_GRAY2BGR);

        cv::Mat result;
        cv::bitwise_and(quantised, edgesColor, result);
        return result;
    }

    cv::Mat applyEdge(const cv::Mat& frameBgr, float threshold)
    {
        cv::Mat gray;
        cv::cvtColor(frameBgr, gray, cv::COLOR_BGR2GRAY);

        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.0);

        const double lower = std::max(0.0f, threshold) * 100.0;
        const double upper = lower * 2.5;

        cv::Mat edges;
        cv::Canny(blurred, edges, lower, upper);

        cv::Mat edgesBgr;
        cv::cvtColor(edges, edgesBgr, cv::COLOR_GRAY2BGR);
        return edgesBgr;
    }
}

namespace FrameProcessor
{
    cv::Mat applyFilter(const cv::Mat& frameBgr,
                        FilterType filter,
                        const FilterParameters& params)
    {
        switch (filter)
        {
            case FilterType::Pixelate:
                return applyPixelate(frameBgr, std::max(1, params.pixelate.blockSize));

            case FilterType::Comic:
                return applyComic(frameBgr, params.comic);

            case FilterType::Edge:
                return applyEdge(frameBgr, params.edge.threshold);

            case FilterType::None:
            default:
                return frameBgr.clone();
        }
    }

    cv::Mat computeAffineMatrix(const cv::Size& frameSize,
                                const TransformParams& transform)
    {
        const cv::Point2f centre(frameSize.width * 0.5f,
                                 frameSize.height * 0.5f);

        cv::Mat rotation = cv::getRotationMatrix2D(
            centre, transform.rotationDegrees, transform.scale);

        rotation.at<double>(0, 2) += transform.translateX;
        rotation.at<double>(1, 2) += transform.translateY;

        cv::Mat rotation32f;
        rotation.convertTo(rotation32f, CV_32F);
        return rotation32f;
    }

    void applyTransform(cv::Mat& frameBgr,
                        const TransformParams& transform)
    {
        if (!transform.isActive())
        {
            return;
        }

        cv::Mat matrix = computeAffineMatrix(frameBgr.size(), transform);
        cv::warpAffine(frameBgr, frameBgr, matrix, frameBgr.size(),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));
    }
}
