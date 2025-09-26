#include "stitch.hpp"
#include "features.hpp"
#include "matching.hpp"
#include "homography.hpp"
#include "warp.hpp"
#include "preprocess.hpp"
#include "blend.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace vc {
static KPDesc runDetector(const cv::Mat& img, Detector d) {
    switch (d) {
        case Detector::SIFT: return detectSIFT(img);
        case Detector::ORB: return detectORB(img);
        case Detector::AKAZE: return detectAKAZE(img);
        default: return detectORB(img);
    }
}

static Distance distTypeFor(Detector d) {
    return d == Detector::ORB ? Distance::HAMMING : Distance::L2;
}

cv::Mat stitchImages(const std::vector<cv::Mat>& imgs,
                     Detector detector,
                     vc::BlendMode blendMode,
                     int ransacIter,
                     double reprojThresh,
                     double ratio,
                     bool debug,
                     const std::string& outDir) {
    if (imgs.empty()) return cv::Mat();
    // Prepare output directory
    std::filesystem::create_directories(outDir);
    // Save params
    {
        std::ofstream ofs(outDir + "/params.txt");
        ofs << "detector=" << (detector==Detector::SIFT?"sift":detector==Detector::ORB?"orb":"akaze") << "\n";
        ofs << "blend=" << (blendMode==BlendMode::OVERLAY?"overlay":"feather") << "\n";
        ofs << "ratio=" << ratio << "\n";
        ofs << "ransac_iter=" << ransacIter << "\n";
        ofs << "reproj_th=" << reprojThresh << "\n";
        ofs << "debug=" << (debug?1:0) << "\n";
        ofs.flush();
    }

    cv::Mat pano = imgs[0].clone();

    for (size_t i = 1; i < imgs.size(); ++i) {
        // Debug outputs
        char namebuf[256];

        std::cout << "[" << i << "/" << imgs.size()-1 << "] Detect features..." << std::endl;
        KPDesc a = runDetector(pano, detector);
        KPDesc b = runDetector(imgs[i], detector);
        std::cout << "  pano kps=" << a.kps.size() << ", new kps=" << b.kps.size() << std::endl;

        auto distType = distTypeFor(detector);
        std::cout << "  Match descriptors..." << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto knn = bruteForceMatchKNN(a.desc, b.desc, distType, 2);
        std::vector<Match> good = ratioTest(knn, ratio);
        auto t1 = std::chrono::high_resolution_clock::now();
        double matchMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  good matches=" << good.size() << ", time(ms)=" << matchMs << std::endl;

        if (debug) {
            cv::Mat imgKP1, imgKP2;
            cv::drawKeypoints(pano, a.kps, imgKP1, cv::Scalar(0,255,0));
            cv::drawKeypoints(imgs[i], b.kps, imgKP2, cv::Scalar(0,255,0));
            snprintf(namebuf, sizeof(namebuf), "%s/kps_%zu_a.jpg", outDir.c_str(), i);
            cv::imwrite(namebuf, imgKP1);
            snprintf(namebuf, sizeof(namebuf), "%s/kps_%zu_b.jpg", outDir.c_str(), i);
            cv::imwrite(namebuf, imgKP2);

            std::vector<cv::DMatch> dm; dm.reserve(good.size());
            for (size_t t = 0; t < good.size(); ++t) dm.emplace_back(good[t].queryIdx, good[t].trainIdx, static_cast<float>(good[t].dist));
            cv::Mat matchesImg;
            cv::drawMatches(pano, a.kps, imgs[i], b.kps, dm, matchesImg);
            snprintf(namebuf, sizeof(namebuf), "%s/matches_%zu.jpg", outDir.c_str(), i);
            cv::imwrite(namebuf, matchesImg);
        }

        std::vector<cv::Point2f> srcPts, dstPts;
        srcPts.reserve(good.size());
        dstPts.reserve(good.size());
        for (const auto& m : good) {
            srcPts.push_back(a.kps[m.queryIdx].pt);
            dstPts.push_back(b.kps[m.trainIdx].pt);
        }
        std::vector<unsigned char> mask_p2n, mask_n2p;
        std::cout << "  RANSAC homography..." << std::endl;
        cv::Mat H_p2n = ransacHomography(srcPts, dstPts, ransacIter, reprojThresh, mask_p2n); // pano->new
        cv::Mat H_n2p = ransacHomography(dstPts, srcPts, ransacIter, reprojThresh, mask_n2p); // new->pano
        if ((H_p2n.empty() || !cv::checkRange(H_p2n)) && (H_n2p.empty() || !cv::checkRange(H_n2p))) return pano;

        // Choose direction: prefer the one with more inliers
        int in_p2n = 0, in_n2p = 0;
        for (auto v : mask_p2n) in_p2n += (v ? 1 : 0);
        for (auto v : mask_n2p) in_n2p += (v ? 1 : 0);
        bool use_n2p = in_n2p >= in_p2n;
        std::cout << "  inliers(p2n)=" << in_p2n << ", inliers(n2p)=" << in_n2p << ", use=" << (use_n2p?"n2p":"p2n") << std::endl;
        cv::Mat H_new_to_pano = use_n2p ? H_n2p : H_p2n.inv();
        if (H_new_to_pano.empty() || !cv::checkRange(H_new_to_pano)) return pano;

        if (debug) {
            const auto& maskUse = use_n2p ? mask_n2p : mask_p2n;
            std::vector<cv::DMatch> dmIn;
            std::vector<cv::KeyPoint> a_in, b_in; a_in.reserve(maskUse.size()); b_in.reserve(maskUse.size());
            for (size_t t = 0; t < maskUse.size(); ++t) {
                if (maskUse[t]) {
                    a_in.push_back(a.kps[good[t].queryIdx]);
                    b_in.push_back(b.kps[good[t].trainIdx]);
                    dmIn.emplace_back(static_cast<int>(a_in.size()-1), static_cast<int>(b_in.size()-1), 0.f);
                }
            }
            if (!a_in.empty()) {
                cv::Mat inlierImg;
                cv::drawMatches(pano, a_in, imgs[i], b_in, dmIn, inlierImg);
                snprintf(namebuf, sizeof(namebuf), "%s/inliers_%zu.jpg", outDir.c_str(), i);
                cv::imwrite(namebuf, inlierImg);
            }
        }

        // Compute dynamic canvas by transforming corners
        std::vector<cv::Point2f> panoCorners = {
            cv::Point2f(0, 0),
            cv::Point2f(static_cast<float>(pano.cols), 0),
            cv::Point2f(static_cast<float>(pano.cols), static_cast<float>(pano.rows)),
            cv::Point2f(0, static_cast<float>(pano.rows))
        };

        std::vector<cv::Point2f> imgCorners = {
            cv::Point2f(0, 0),
            cv::Point2f(static_cast<float>(imgs[i].cols), 0),
            cv::Point2f(static_cast<float>(imgs[i].cols), static_cast<float>(imgs[i].rows)),
            cv::Point2f(0, static_cast<float>(imgs[i].rows))
        };

        // H maps pano -> img; we need new->pano for corner transform
        // H_new_to_pano chosen above
        std::vector<cv::Point2f> imgCornersWarped;
        cv::perspectiveTransform(imgCorners, imgCornersWarped, H_new_to_pano);

        float minX = 0.f, minY = 0.f, maxX = static_cast<float>(pano.cols), maxY = static_cast<float>(pano.rows);
        for (const auto& p : imgCornersWarped) {
            minX = std::min(minX, p.x);
            minY = std::min(minY, p.y);
            maxX = std::max(maxX, p.x);
            maxY = std::max(maxY, p.y);
        }

        int tx = (minX < 0.f) ? static_cast<int>(std::floor(-minX)) : 0;
        int ty = (minY < 0.f) ? static_cast<int>(std::floor(-minY)) : 0;
        int outW = static_cast<int>(std::ceil(maxX + tx));
        int outH = static_cast<int>(std::ceil(maxY + ty));

        // Compose translation so that everything is positive in the canvas
        cv::Mat T = (cv::Mat_<double>(3,3) << 1, 0, tx, 0, 1, ty, 0, 0, 1);
        cv::Mat G = T * H_new_to_pano; // pass G; warper will invert internally

        std::cout << "  Warp new image... outW=" << outW << ", outH=" << outH << std::endl;
        cv::Mat warped = warpPerspectiveCustom(imgs[i], G, cv::Size(outW, outH));

        cv::Mat canvas(outH, outW, CV_8UC3, cv::Scalar::all(0));
        pano.copyTo(canvas(cv::Rect(tx, ty, pano.cols, pano.rows)));

        // Mask for blending
        cv::Mat nonzeroGray, mask;
        cv::cvtColor(warped, nonzeroGray, cv::COLOR_BGR2GRAY);
        cv::threshold(nonzeroGray, mask, 1, 255, cv::THRESH_BINARY);

        cv::Mat blended;
        if (blendMode == BlendMode::OVERLAY) {
            blended = blendOverlay(canvas, warped, mask);
        } else {
            // Robust feathering: weights from distance transforms of both masks
            cv::Mat maskBase = cv::Mat::zeros(canvas.size(), CV_8U);
            cv::rectangle(maskBase, cv::Rect(tx, ty, pano.cols, pano.rows), cv::Scalar(255), cv::FILLED);

            cv::Mat dtTop, dtBase;
            cv::distanceTransform(mask, dtTop, cv::DIST_L2, 3);
            cv::distanceTransform(maskBase, dtBase, cv::DIST_L2, 3);

            cv::Mat wTop, wBase, denom;
            dtTop.convertTo(wTop, CV_32F);
            dtBase.convertTo(wBase, CV_32F);
            denom = wTop + wBase + 1e-6f;
            cv::divide(wTop, denom, wTop);
            // wBase = 1 - wTop
            wBase = 1.0f - wTop;

            cv::Mat baseF, topF;
            canvas.convertTo(baseF, CV_32F, 1.0/255.0);
            warped.convertTo(topF, CV_32F, 1.0/255.0);

            std::vector<cv::Mat> bc, tc, oc;
            cv::split(baseF, bc);
            cv::split(topF, tc);
            oc.resize(3);
            for (int c = 0; c < 3; ++c) {
                oc[c] = bc[c].mul(wBase) + tc[c].mul(wTop);
            }
            cv::Mat outF; cv::merge(oc, outF);
            outF.convertTo(blended, CV_8U, 255.0);
        }

        pano = blended;
    }

    // Auto-crop black borders
    cv::Mat gray, mask;
    std::cout << "Auto-crop..." << std::endl;
    cv::cvtColor(pano, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 1, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> pts; cv::findNonZero(mask, pts);
    if (!pts.empty()) {
        cv::Rect roi = cv::boundingRect(pts);
        pano = pano(roi).clone();
    }

    return pano;
}
}
