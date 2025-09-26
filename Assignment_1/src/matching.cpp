#include "matching.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <cmath>
#include <algorithm>

namespace vc {
static inline double squaredL2(const float* a, const float* b, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return s;
}

double euclideanDistance(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.cols == b.cols && a.type() == b.type());
    CV_Assert(a.rows == 1 && b.rows == 1);
    if (a.type() == CV_32F) {
        return std::sqrt(squaredL2(a.ptr<float>(), b.ptr<float>(), a.cols));
    }
    cv::Mat af, bf;
    a.convertTo(af, CV_32F);
    b.convertTo(bf, CV_32F);
    return std::sqrt(squaredL2(af.ptr<float>(), bf.ptr<float>(), af.cols));
}

static inline int popcount32(uint32_t x) {
    return __builtin_popcount(x);
}

int hammingDistance(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.cols == b.cols && a.type() == b.type());
    CV_Assert(a.rows == 1 && b.rows == 1);
    CV_Assert(a.type() == CV_8U);
    const uint8_t* ap = a.ptr<uint8_t>();
    const uint8_t* bp = b.ptr<uint8_t>();
    int dist = 0;
    int bytes = a.cols;
    int i = 0;
    for (; i + 4 <= bytes; i += 4) {
        uint32_t av = *reinterpret_cast<const uint32_t*>(ap + i);
        uint32_t bv = *reinterpret_cast<const uint32_t*>(bp + i);
        dist += popcount32(av ^ bv);
    }
    for (; i < bytes; ++i) dist += __builtin_popcount((unsigned)(ap[i] ^ bp[i]));
    return dist;
}

std::vector<Match> bruteForceMatch(const cv::Mat& desc1, const cv::Mat& desc2, Distance distType) {
    std::vector<Match> matches;
    if (desc1.empty() || desc2.empty()) return matches;
    const int n1 = desc1.rows;
    const int n2 = desc2.rows;

    for (int i = 0; i < n1; ++i) {
        double best = std::numeric_limits<double>::infinity();
        int bestIdx = -1;
        for (int j = 0; j < n2; ++j) {
            double d = 0.0;
            if (distType == Distance::L2) {
                d = euclideanDistance(desc1.row(i), desc2.row(j));
            } else {
                d = static_cast<double>(hammingDistance(desc1.row(i), desc2.row(j)));
            }
            if (d < best) { best = d; bestIdx = j; }
        }
        if (bestIdx >= 0) matches.push_back({i, bestIdx, best});
    }
    return matches;
}

std::vector<std::pair<Match, Match>> bruteForceMatchKNN(const cv::Mat& desc1, const cv::Mat& desc2, Distance distType, int k) {
    std::vector<std::pair<Match, Match>> knn;
    if (desc1.empty() || desc2.empty() || k < 2) return knn;
    const int n1 = desc1.rows;
    const int n2 = desc2.rows;

    for (int i = 0; i < n1; ++i) {
        double best = std::numeric_limits<double>::infinity();
        double second = std::numeric_limits<double>::infinity();
        int bestIdx = -1, secondIdx = -1;
        for (int j = 0; j < n2; ++j) {
            double d = 0.0;
            if (distType == Distance::L2) {
                d = euclideanDistance(desc1.row(i), desc2.row(j));
            } else {
                d = static_cast<double>(hammingDistance(desc1.row(i), desc2.row(j)));
            }
            if (d < best) {
                second = best; secondIdx = bestIdx;
                best = d; bestIdx = j;
            } else if (d < second) {
                second = d; secondIdx = j;
            }
        }
        if (bestIdx >= 0 && secondIdx >= 0) {
            knn.push_back({ {i, bestIdx, best}, {i, secondIdx, second} });
        }
    }
    return knn;
}

std::vector<Match> ratioTest(const std::vector<std::pair<Match,Match>>& knn, double ratio) {
    std::vector<Match> good;
    for (const auto& p : knn) {
        const auto& m1 = p.first;
        const auto& m2 = p.second;
        if (m2.dist <= 1e-12) continue;
        if (m1.dist / m2.dist < ratio) good.push_back(m1);
    }
    return good;
}
}
