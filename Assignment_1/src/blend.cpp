#include "blend.hpp"
#include <opencv2/imgproc.hpp>

namespace vc {
cv::Mat blendOverlay(const cv::Mat& baseImg, const cv::Mat& topImg, const cv::Mat& mask) {
    CV_Assert(baseImg.type() == CV_8UC3 && topImg.type() == CV_8UC3);
    CV_Assert(baseImg.size() == topImg.size());
    cv::Mat out = baseImg.clone();
    if (!mask.empty()) {
        topImg.copyTo(out, mask);
    } else {
        out = topImg.clone();
    }
    return out;
}

cv::Mat blendFeather(const cv::Mat& baseImg, const cv::Mat& topImg, const cv::Mat& weightMask, double eps) {
    CV_Assert(baseImg.type() == CV_8UC3 && topImg.type() == CV_8UC3);
    CV_Assert(baseImg.size() == topImg.size());

    cv::Mat w;
    if (weightMask.empty()) {
        w = cv::Mat(baseImg.size(), CV_32F, cv::Scalar(0.5f));
    } else {
        if (weightMask.type() == CV_8U) {
            weightMask.convertTo(w, CV_32F, 1.0/255.0);
        } else {
            w = weightMask;
            w.convertTo(w, CV_32F);
        }
    }

    cv::Mat baseF, topF;
    baseImg.convertTo(baseF, CV_32F, 1.0/255.0);
    topImg.convertTo(topF, CV_32F, 1.0/255.0);

    std::vector<cv::Mat> bc, tc, oc;
    cv::split(baseF, bc);
    cv::split(topF, tc);

    cv::Mat wInv = 1.0 - w;
    for (int c = 0; c < 3; ++c) {
        cv::Mat ch = bc[c].mul(wInv) + tc[c].mul(w);
        oc.push_back(ch);
    }

    cv::Mat outF;
    cv::merge(oc, outF);
    cv::Mat out;
    outF.convertTo(out, CV_8U, 255.0);
    return out;
}
}
