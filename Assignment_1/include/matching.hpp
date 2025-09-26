#pragma once
#include <opencv2/core.hpp>
#include <vector>
namespace vc {
enum class Distance { L2, HAMMING };

// Distances operate on single descriptor rows
double euclideanDistance(const cv::Mat& a, const cv::Mat& b);
int hammingDistance(const cv::Mat& a, const cv::Mat& b);

struct Match { int queryIdx; int trainIdx; double dist; };

// Return best match per query (1-NN), and optionally a 2-NN set for ratio test
std::vector<Match> bruteForceMatch(const cv::Mat& desc1, const cv::Mat& desc2, Distance distType);
std::vector<std::pair<Match, Match>> bruteForceMatchKNN(const cv::Mat& desc1, const cv::Mat& desc2, Distance distType, int k);

std::vector<Match> ratioTest(const std::vector<std::pair<Match,Match>>& knn, double ratio);
}
