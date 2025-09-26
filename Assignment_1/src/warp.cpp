#include "warp.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cmath>

namespace vc {
static inline bool inBounds(int x, int y, int w, int h) {
    return x >= 0 && x < w && y >= 0 && y < h;
}

static cv::Vec3f bilinearAt(const cv::Mat& src, float x, float y) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = x - x0;
    float ay = y - y0;
    cv::Vec3f c00 = inBounds(x0, y0, src.cols, src.rows) ? src.at<cv::Vec3b>(y0, x0) : cv::Vec3b(0,0,0);
    cv::Vec3f c10 = inBounds(x1, y0, src.cols, src.rows) ? src.at<cv::Vec3b>(y0, x1) : cv::Vec3b(0,0,0);
    cv::Vec3f c01 = inBounds(x0, y1, src.cols, src.rows) ? src.at<cv::Vec3b>(y1, x0) : cv::Vec3b(0,0,0);
    cv::Vec3f c11 = inBounds(x1, y1, src.cols, src.rows) ? src.at<cv::Vec3b>(y1, x1) : cv::Vec3b(0,0,0);
    cv::Vec3f c0 = c00 * (1.0f - ax) + c10 * ax;
    cv::Vec3f c1 = c01 * (1.0f - ax) + c11 * ax;
    cv::Vec3f c = c0 * (1.0f - ay) + c1 * ay;
    return c;
}

cv::Mat warpPerspectiveCustom(const cv::Mat& src, const cv::Mat& H, cv::Size outSize) {
    CV_Assert(src.type() == CV_8UC3);
    cv::Mat Hinv = H.inv();
    cv::Mat dst(outSize, CV_8UC3, cv::Scalar::all(0));

    for (int y = 0; y < outSize.height; ++y) {
        for (int x = 0; x < outSize.width; ++x) {
            cv::Vec3d p(x, y, 1.0);
            cv::Vec3d q = cv::Mat(Hinv * cv::Mat(p));
            float sx = static_cast<float>(q[0] / q[2]);
            float sy = static_cast<float>(q[1] / q[2]);
            if (sx >= -1 && sy >= -1 && sx < src.cols && sy < src.rows) {
                cv::Vec3f c = bilinearAt(src, sx, sy);
                dst.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(c[0]),
                                                    cv::saturate_cast<uchar>(c[1]),
                                                    cv::saturate_cast<uchar>(c[2]));
            }
        }
    }
    return dst;
}
}
