#include "homography.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <random>
#include <numeric>
#include <unordered_set>

namespace vc {
static cv::Mat buildA(const std::vector<cv::Point2f>& srcPts, const std::vector<cv::Point2f>& dstPts) {
    const int n = static_cast<int>(srcPts.size());
    cv::Mat A(2*n, 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        double x = srcPts[i].x, y = srcPts[i].y;
        double u = dstPts[i].x, v = dstPts[i].y;
        A.at<double>(2*i, 0) = -x;
        A.at<double>(2*i, 1) = -y;
        A.at<double>(2*i, 2) = -1;
        A.at<double>(2*i, 3) = 0;
        A.at<double>(2*i, 4) = 0;
        A.at<double>(2*i, 5) = 0;
        A.at<double>(2*i, 6) = x*u;
        A.at<double>(2*i, 7) = y*u;
        A.at<double>(2*i, 8) = u;

        A.at<double>(2*i+1, 0) = 0;
        A.at<double>(2*i+1, 1) = 0;
        A.at<double>(2*i+1, 2) = 0;
        A.at<double>(2*i+1, 3) = -x;
        A.at<double>(2*i+1, 4) = -y;
        A.at<double>(2*i+1, 5) = -1;
        A.at<double>(2*i+1, 6) = x*v;
        A.at<double>(2*i+1, 7) = y*v;
        A.at<double>(2*i+1, 8) = v;
    }
    return A;
}

cv::Mat computeHomographyDLT(const std::vector<cv::Point2f>& srcPts, const std::vector<cv::Point2f>& dstPts) {
    CV_Assert(srcPts.size() == dstPts.size());
    CV_Assert(srcPts.size() >= 4);

    // Hartley normalization for numerical stability
    auto normalize = [](const std::vector<cv::Point2f>& pts, cv::Mat& T) {
        cv::Point2d mean(0,0);
        for (auto& p : pts) { mean.x += p.x; mean.y += p.y; }
        mean.x /= pts.size(); mean.y /= pts.size();
        double avgDist = 0.0;
        for (auto& p : pts) {
            double dx = p.x - mean.x, dy = p.y - mean.y;
            avgDist += std::sqrt(dx*dx + dy*dy);
        }
        avgDist /= pts.size();
        double s = (avgDist > 0) ? std::sqrt(2.0) / avgDist : 1.0;
        T = (cv::Mat_<double>(3,3) << s, 0, -s*mean.x, 0, s, -s*mean.y, 0, 0, 1);
        std::vector<cv::Point2f> out; out.reserve(pts.size());
        for (auto& p : pts) {
            cv::Vec3d ph(p.x, p.y, 1.0);
            cv::Vec3d q = cv::Mat(T * cv::Mat(ph));
            out.emplace_back(static_cast<float>(q[0]/q[2]), static_cast<float>(q[1]/q[2]));
        }
        return out;
    };

    cv::Mat Tsrc, Tdst;
    std::vector<cv::Point2f> nsrc = normalize(srcPts, Tsrc);
    std::vector<cv::Point2f> ndst = normalize(dstPts, Tdst);

    cv::Mat A = buildA(nsrc, ndst);
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);
    cv::Mat Hn = vt.row(vt.rows - 1).reshape(0, 3);
    // Denormalize: H = Tdst^{-1} * Hn * Tsrc
    cv::Mat H = Tdst.inv() * Hn * Tsrc;
    H /= H.at<double>(2, 2);
    return H;
}

static double reprojError(const cv::Mat& H, const cv::Point2f& p, const cv::Point2f& q) {
    cv::Vec3d ph(p.x, p.y, 1.0);
    cv::Vec3d wh = cv::Mat(H * cv::Mat(ph));
    double wx = wh[0] / wh[2];
    double wy = wh[1] / wh[2];
    double dx = wx - q.x;
    double dy = wy - q.y;
    return std::sqrt(dx*dx + dy*dy);
}

cv::Mat ransacHomography(const std::vector<cv::Point2f>& srcPts,
                         const std::vector<cv::Point2f>& dstPts,
                         int iterations, double thresh,
                         std::vector<unsigned char>& inlierMask) {
    CV_Assert(srcPts.size() == dstPts.size());
    const int n = static_cast<int>(srcPts.size());
    inlierMask.assign(n, 0);
    if (n < 4) return cv::Mat();

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> uni(0, n - 1);

    int bestInliers = -1;
    cv::Mat bestH;

    std::vector<int> idx(4);
    for (int it = 0; it < iterations; ++it) {
        // sample 4 unique indices
        std::unordered_set<int> used;
        int k = 0;
        while (k < 4) {
            int r = uni(rng);
            if (used.insert(r).second) {
                idx[k++] = r;
            }
        }
        std::vector<cv::Point2f> s(4), d(4);
        for (int t = 0; t < 4; ++t) { s[t] = srcPts[idx[t]]; d[t] = dstPts[idx[t]]; }
        cv::Mat H = computeHomographyDLT(s, d);
        int inliers = 0;
        for (int i = 0; i < n; ++i) {
            double e = reprojError(H, srcPts[i], dstPts[i]);
            if (e < thresh) ++inliers;
        }
        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestH = H.clone();
        }
    }

    if (bestH.empty()) return bestH;

    // Build final mask
    for (int i = 0; i < n; ++i) {
        double e = reprojError(bestH, srcPts[i], dstPts[i]);
        inlierMask[i] = (e < thresh) ? 1 : 0;
    }

    // Optional: refine using inliers
    std::vector<cv::Point2f> sIn, dIn;
    sIn.reserve(bestInliers);
    dIn.reserve(bestInliers);
    for (int i = 0; i < n; ++i) if (inlierMask[i]) { sIn.push_back(srcPts[i]); dIn.push_back(dstPts[i]); }
    if (sIn.size() >= 4) {
        bestH = computeHomographyDLT(sIn, dIn);
    }

    return bestH;
}
}
