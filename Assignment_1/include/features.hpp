#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
namespace vc {
struct KPDesc { std::vector<cv::KeyPoint> kps; cv::Mat desc; };
KPDesc detectSIFT(const cv::Mat& img);
KPDesc detectORB(const cv::Mat& img);
KPDesc detectAKAZE(const cv::Mat& img);
}
