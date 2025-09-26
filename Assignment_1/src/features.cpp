#include "features.hpp"
#include "preprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

namespace vc {
static KPDesc detectWith(cv::Ptr<cv::Feature2D> det, const cv::Mat& img) {
    KPDesc out;
    if (img.empty() || det.empty()) return out;
    cv::Mat gray = toGray(img);
    det->detectAndCompute(gray, cv::noArray(), out.kps, out.desc);
    return out;
}

KPDesc detectSIFT(const cv::Mat& img) {
    cv::Ptr<cv::Feature2D> det;
    try {
#if CV_VERSION_MAJOR >= 4
        det = cv::SIFT::create();
#else
        det = cv::xfeatures2d::SIFT::create();
#endif
    } catch (...) {
        det = cv::ORB::create(5000);
    }
    return detectWith(det, img);
}

KPDesc detectORB(const cv::Mat& img) {
    auto det = cv::ORB::create(5000);
    return detectWith(det, img);
}

KPDesc detectAKAZE(const cv::Mat& img) {
    auto det = cv::AKAZE::create();
    return detectWith(det, img);
}
}
