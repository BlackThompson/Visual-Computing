#include "preprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace vc {
cv::Mat toGray(const cv::Mat& src) {
    if (src.empty()) return cv::Mat();
    if (src.channels() == 1) return src.clone();
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat normalizeImage(const cv::Mat& src) {
    if (src.empty()) return cv::Mat();
    cv::Mat srcFloat;
    if (src.depth() == CV_32F) {
        srcFloat = src;
    } else {
        src.convertTo(srcFloat, CV_32F);
    }
    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(srcFloat, &minVal, &maxVal);
    if (maxVal - minVal < 1e-12) {
        cv::Mat zeros = cv::Mat::zeros(src.size(), CV_32F);
        return zeros;
    }
    cv::Mat norm = (srcFloat - static_cast<float>(minVal)) / static_cast<float>(maxVal - minVal);
    return norm;
}

cv::Mat gaussianBlur(const cv::Mat& src, int ksize, double sigma) {
    if (src.empty()) return cv::Mat();
    if (ksize <= 1) return src.clone();
    if (ksize % 2 == 0) ksize += 1; // ensure odd
    cv::Mat dst;
    cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), sigma, sigma, cv::BORDER_REPLICATE);
    return dst;
}
}
