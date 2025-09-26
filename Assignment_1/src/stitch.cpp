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
#include <numeric>

namespace vc {
static void writeCsvRow(const std::string& filePath, const std::string& header, const std::string& row) {
    const bool exists = std::filesystem::exists(filePath);
    std::ofstream ofs(filePath, std::ios::app);
    if (!exists) ofs << header << "\n";
    ofs << row << "\n";
}

static std::string toString(Detector d) {
    return d==Detector::SIFT?"sift":d==Detector::ORB?"orb":"akaze";
}

static std::string toString(BlendMode b) {
    return b==BlendMode::OVERLAY?"overlay":"feather";
}
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
                     const std::string& outDir,
                     const std::string& setId,
                     const std::string& pairId) {
    if (imgs.empty()) return cv::Mat();
    // Prepare output directory
    std::filesystem::create_directories(outDir);
    std::string vizRoot = outDir;
    if (!setId.empty() && !pairId.empty()) {
        vizRoot = outDir + "/viz/" + setId + "/" + toString(detector) + "/" + pairId;
        std::filesystem::create_directories(vizRoot);
    }
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
        // Build detector ptr
        cv::Ptr<cv::Feature2D> detPtr;
        if (detector == Detector::SIFT) detPtr = cv::SIFT::create();
        else if (detector == Detector::ORB) detPtr = cv::ORB::create(5000);
        else detPtr = cv::AKAZE::create();

        // Detect + describe with timing (pano)
        KPDesc a; KPDesc b;
        auto t_d0 = std::chrono::high_resolution_clock::now();
        std::vector<cv::KeyPoint> a_kps; detPtr->detect(pano, a_kps);
        auto t_d1 = std::chrono::high_resolution_clock::now();
        cv::Mat a_desc; detPtr->compute(pano, a_kps, a_desc);
        auto t_d2 = std::chrono::high_resolution_clock::now();
        a.kps = std::move(a_kps); a.desc = a_desc;

        // Detect + describe with timing (new)
        auto t_e0 = std::chrono::high_resolution_clock::now();
        std::vector<cv::KeyPoint> b_kps; detPtr->detect(imgs[i], b_kps);
        auto t_e1 = std::chrono::high_resolution_clock::now();
        cv::Mat b_desc; detPtr->compute(imgs[i], b_kps, b_desc);
        auto t_e2 = std::chrono::high_resolution_clock::now();
        b.kps = std::move(b_kps); b.desc = b_desc;

        double pano_detect_ms = std::chrono::duration<double, std::milli>(t_d1 - t_d0).count();
        double pano_desc_ms   = std::chrono::duration<double, std::milli>(t_d2 - t_d1).count();
        double new_detect_ms  = std::chrono::duration<double, std::milli>(t_e1 - t_e0).count();
        double new_desc_ms    = std::chrono::duration<double, std::milli>(t_e2 - t_e1).count();

        auto avgStats = [](const std::vector<cv::KeyPoint>& kps){
            double avgSize = 0.0, avgResp = 0.0; size_t n = kps.size();
            for (const auto& k : kps) { avgSize += k.size; avgResp += k.response; }
            if (n>0) { avgSize/=n; avgResp/=n; }
            return std::pair<double,double>(avgSize, avgResp);
        };
        auto [pano_avg_size, pano_avg_resp] = avgStats(a.kps);
        auto [new_avg_size, new_avg_resp]   = avgStats(b.kps);

        std::cout << "  pano kps=" << a.kps.size() << ", new kps=" << b.kps.size() << std::endl;
        // Log detect/describe per image
        const std::string ddHead = "run_id,detector,image_role,num_keypoints,detect_time_ms,describe_time_ms,avg_keypoint_scale,avg_response";
        const std::string run_id = outDir.substr(outDir.find_last_of('/')+1);
        char rowbuf[512];
        std::snprintf(rowbuf, sizeof(rowbuf), "%s,%s,pano,%zu,%.3f,%.3f,%.3f,%.3f",
                      run_id.c_str(), toString(detector).c_str(), a.kps.size(), pano_detect_ms, pano_desc_ms, pano_avg_size, pano_avg_resp);
        writeCsvRow(outDir + "/detect_describe.csv", ddHead, rowbuf);
        std::snprintf(rowbuf, sizeof(rowbuf), "%s,%s,new,%zu,%.3f,%.3f,%.3f,%.3f",
                      run_id.c_str(), toString(detector).c_str(), b.kps.size(), new_detect_ms, new_desc_ms, new_avg_size, new_avg_resp);
        writeCsvRow(outDir + "/detect_describe.csv", ddHead, rowbuf);

        auto distType = distTypeFor(detector);
        std::cout << "  Match descriptors..." << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t_m0 = std::chrono::high_resolution_clock::now();
        auto knn = bruteForceMatchKNN(a.desc, b.desc, distType, 2);
        auto t_m1 = std::chrono::high_resolution_clock::now();
        std::vector<Match> good = ratioTest(knn, ratio);
        auto t_m2 = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();
        double matchMs = std::chrono::duration<double, std::milli>(t_m1 - t_m0).count();
        double filterMs = std::chrono::duration<double, std::milli>(t_m2 - t_m1).count();
        std::cout << "  good matches=" << good.size() << ", time(ms)=" << (matchMs+filterMs) << std::endl;

        // Distances for histograms
        std::vector<double> raw_dists; raw_dists.reserve(knn.size());
        for (const auto& pr : knn) raw_dists.push_back(pr.first.dist);
        std::vector<double> kept_dists; kept_dists.reserve(good.size());
        for (const auto& m : good) kept_dists.push_back(m.dist);
        // Save distances to CSV
        {
            std::ofstream fd(outDir + "/raw_distances.csv");
            for (double d : raw_dists) fd << d << "\n";
        }
        {
            std::ofstream fd(outDir + "/kept_distances.csv");
            for (double d : kept_dists) fd << d << "\n";
        }
        // Matching CSV
        const std::string mHead = "run_id,detector,knn_k,raw_matches,raw_match_time_ms,ratio,kept_matches,filter_time_ms,dist_mean,dist_std";
        auto meanStd = [](const std::vector<double>& v){
            if (v.empty()) return std::pair<double,double>(0.0,0.0);
            double s = std::accumulate(v.begin(), v.end(), 0.0);
            double mean = s / v.size();
            double var = 0.0; for (double x : v){ double d=x-mean; var += d*d; }
            var /= v.size();
            return std::pair<double,double>(mean, std::sqrt(var));
        };
        auto [dm, ds] = meanStd(kept_dists);
        std::snprintf(rowbuf, sizeof(rowbuf), "%s,%s,%d,%zu,%.3f,%.2f,%zu,%.3f,%.6f,%.6f",
                      run_id.c_str(), toString(detector).c_str(), 2, knn.size(), matchMs, ratio, good.size(), filterMs, dm, ds);
        writeCsvRow(outDir + "/matching.csv", mHead, rowbuf);

        if (debug) {
            // 1) More visible keypoints: rich style + bright color
            cv::Mat imgKP1, imgKP2;
            cv::drawKeypoints(pano, a.kps, imgKP1, cv::Scalar(0,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::drawKeypoints(imgs[i], b.kps, imgKP2, cv::Scalar(0,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            snprintf(namebuf, sizeof(namebuf), "%s/kps_%zu_a.jpg", vizRoot.c_str(), i);
            cv::imwrite(namebuf, imgKP1);
            snprintf(namebuf, sizeof(namebuf), "%s/kps_%zu_b.jpg", vizRoot.c_str(), i);
            cv::imwrite(namebuf, imgKP2);

            // 2) Dense matches (OpenCV default)
            std::vector<cv::DMatch> dm; dm.reserve(good.size());
            for (size_t t = 0; t < good.size(); ++t) dm.emplace_back(good[t].queryIdx, good[t].trainIdx, static_cast<float>(good[t].dist));
            cv::Mat matchesImg;
            cv::drawMatches(pano, a.kps, imgs[i], b.kps, dm, matchesImg);
            snprintf(namebuf, sizeof(namebuf), "%s/matches_%zu.jpg", vizRoot.c_str(), i);
            cv::imwrite(namebuf, matchesImg);

            // 3) Annotated matches: colored pairs + pair index labels (topN to avoid clutter)
            auto colorFromIndex = [](int idx){
                int hue = (idx * 37) % 180; // spread hues
                cv::Mat hsv(1,1,CV_8UC3, cv::Vec3b((uchar)hue, 200, 255));
                cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
                cv::Vec3b c = bgr.at<cv::Vec3b>(0,0);
                return cv::Scalar(c[0], c[1], c[2]);
            };
            int H = std::max(pano.rows, imgs[i].rows);
            int W = pano.cols + imgs[i].cols;
            cv::Mat anno(H, W, CV_8UC3, cv::Scalar::all(0));
            pano.copyTo(anno(cv::Rect(0, 0, pano.cols, pano.rows)));
            imgs[i].copyTo(anno(cv::Rect(pano.cols, 0, imgs[i].cols, imgs[i].rows)));
            int xOffset = pano.cols;
            int topN = std::min<int>(static_cast<int>(good.size()), 150);
            for (int t = 0; t < topN; ++t) {
                const auto &m = good[t];
                cv::Point p(cvRound(a.kps[m.queryIdx].pt.x), cvRound(a.kps[m.queryIdx].pt.y));
                cv::Point q(cvRound(b.kps[m.trainIdx].pt.x) + xOffset, cvRound(b.kps[m.trainIdx].pt.y));
                cv::Scalar col = colorFromIndex(t);
                cv::circle(anno, p, 4, col, 2, cv::LINE_AA);
                cv::circle(anno, q, 4, col, 2, cv::LINE_AA);
                cv::line(anno, p, q, col, 1, cv::LINE_AA);
                char txt[16]; std::snprintf(txt, sizeof(txt), "%d", t+1);
                cv::putText(anno, txt, p + cv::Point(5,-5), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
                cv::putText(anno, txt, q + cv::Point(5,-5), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
            }
            snprintf(namebuf, sizeof(namebuf), "%s/matches_annotated_%zu.jpg", vizRoot.c_str(), i);
            cv::imwrite(namebuf, anno);
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
        auto t_r0 = std::chrono::high_resolution_clock::now();
        cv::Mat H_p2n = ransacHomography(srcPts, dstPts, ransacIter, reprojThresh, mask_p2n); // pano->new
        cv::Mat H_n2p = ransacHomography(dstPts, srcPts, ransacIter, reprojThresh, mask_n2p); // new->pano
        auto t_r1 = std::chrono::high_resolution_clock::now();
        if ((H_p2n.empty() || !cv::checkRange(H_p2n)) && (H_n2p.empty() || !cv::checkRange(H_n2p))) return pano;

        // Choose direction: prefer the one with more inliers
        int in_p2n = 0, in_n2p = 0;
        for (auto v : mask_p2n) in_p2n += (v ? 1 : 0);
        for (auto v : mask_n2p) in_n2p += (v ? 1 : 0);
        bool use_n2p = in_n2p >= in_p2n;
        std::cout << "  inliers(p2n)=" << in_p2n << ", inliers(n2p)=" << in_n2p << ", use=" << (use_n2p?"n2p":"p2n") << std::endl;
        cv::Mat H_new_to_pano = use_n2p ? H_n2p : H_p2n.inv();
        if (H_new_to_pano.empty() || !cv::checkRange(H_new_to_pano)) return pano;

        // RANSAC CSV (avg reprojection error on inliers)
        auto reprojAvg = [&](const std::vector<cv::Point2f>& P, const std::vector<cv::Point2f>& Q, const std::vector<unsigned char>& m, const cv::Mat& H){
            double s=0.0; int c=0; for(size_t t=0;t<P.size();++t){ if(!m[t]) continue; cv::Vec3d ph(P[t].x,P[t].y,1.0); cv::Vec3d q = cv::Mat(H*cv::Mat(ph)); double wx=q[0]/q[2]; double wy=q[1]/q[2]; double dx=wx-Q[t].x, dy=wy-Q[t].y; s+=std::sqrt(dx*dx+dy*dy); ++c;} return c>0 ? s/c : 0.0; };
        double ransac_ms = std::chrono::duration<double, std::milli>(t_r1 - t_r0).count();
        int inliers = use_n2p ? in_n2p : in_p2n;
        double inlier_ratio = (good.empty()? 0.0 : static_cast<double>(inliers)/static_cast<double>(good.size()));
        double avg_err = use_n2p ? reprojAvg(dstPts, srcPts, mask_n2p, H_n2p) : reprojAvg(srcPts, dstPts, mask_p2n, H_p2n);
        // H flatten
        double h00=H_new_to_pano.at<double>(0,0), h01=H_new_to_pano.at<double>(0,1), h02=H_new_to_pano.at<double>(0,2);
        double h10=H_new_to_pano.at<double>(1,0), h11=H_new_to_pano.at<double>(1,1), h12=H_new_to_pano.at<double>(1,2);
        double h20=H_new_to_pano.at<double>(2,0), h21=H_new_to_pano.at<double>(2,1), h22=H_new_to_pano.at<double>(2,2);
        const std::string rHead = "run_id,detector,thresh_px,iters,inliers,inlier_ratio,ransac_time_ms,avg_reproj_error_px,h00,h01,h02,h10,h11,h12,h20,h21,h22";
        std::snprintf(rowbuf, sizeof(rowbuf), "%s,%s,%.3f,%d,%d,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                      run_id.c_str(), toString(detector).c_str(), reprojThresh, ransacIter, inliers, inlier_ratio, ransac_ms, avg_err,
                      h00,h01,h02,h10,h11,h12,h20,h21,h22);
        writeCsvRow(outDir + "/ransac.csv", rHead, rowbuf);

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
                // Dense inliers image
                cv::Mat inlierImg;
                cv::drawMatches(pano, a_in, imgs[i], b_in, dmIn, inlierImg);
                snprintf(namebuf, sizeof(namebuf), "%s/inliers_%zu.jpg", vizRoot.c_str(), i);
                cv::imwrite(namebuf, inlierImg);

                // Annotated inliers (limit topN)
                auto colorFromIndex = [](int idx){
                    int hue = (idx * 37) % 180; cv::Mat hsv(1,1,CV_8UC3, cv::Vec3b((uchar)hue, 200, 255));
                    cv::Mat bgr; cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR); cv::Vec3b c = bgr.at<cv::Vec3b>(0,0);
                    return cv::Scalar(c[0], c[1], c[2]);
                };
                int H = std::max(pano.rows, imgs[i].rows);
                int W = pano.cols + imgs[i].cols;
                cv::Mat anno(H, W, CV_8UC3, cv::Scalar::all(0));
                pano.copyTo(anno(cv::Rect(0, 0, pano.cols, pano.rows)));
                imgs[i].copyTo(anno(cv::Rect(pano.cols, 0, imgs[i].cols, imgs[i].rows)));
                int xOffset = pano.cols;
                int topN = std::min<int>(static_cast<int>(a_in.size()), 150);
                for (int t = 0; t < topN; ++t) {
                    cv::Point p(cvRound(a_in[t].pt.x), cvRound(a_in[t].pt.y));
                    cv::Point q(cvRound(b_in[t].pt.x) + xOffset, cvRound(b_in[t].pt.y));
                    cv::Scalar col = colorFromIndex(t);
                    cv::circle(anno, p, 4, col, 2, cv::LINE_AA);
                    cv::circle(anno, q, 4, col, 2, cv::LINE_AA);
                    cv::line(anno, p, q, col, 1, cv::LINE_AA);
                    char txt[16]; std::snprintf(txt, sizeof(txt), "%d", t+1);
                    cv::putText(anno, txt, p + cv::Point(5,-5), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
                    cv::putText(anno, txt, q + cv::Point(5,-5), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
                }
                snprintf(namebuf, sizeof(namebuf), "%s/inliers_annotated_%zu.jpg", vizRoot.c_str(), i);
                cv::imwrite(namebuf, anno);
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
        auto t_w0 = std::chrono::high_resolution_clock::now();
        cv::Mat warped = warpPerspectiveCustom(imgs[i], G, cv::Size(outW, outH));
        auto t_w1 = std::chrono::high_resolution_clock::now();

        cv::Mat canvas(outH, outW, CV_8UC3, cv::Scalar::all(0));
        pano.copyTo(canvas(cv::Rect(tx, ty, pano.cols, pano.rows)));

        // Mask for blending
        cv::Mat nonzeroGray, mask;
        cv::cvtColor(warped, nonzeroGray, cv::COLOR_BGR2GRAY);
        cv::threshold(nonzeroGray, mask, 1, 255, cv::THRESH_BINARY);

        cv::Mat blended;
        double warp_ms = std::chrono::duration<double, std::milli>(t_w1 - t_w0).count();
        double blend_ms = 0.0;
        if (blendMode == BlendMode::OVERLAY) {
            auto t_b0 = std::chrono::high_resolution_clock::now();
            blended = blendOverlay(canvas, warped, mask);
            auto t_b1 = std::chrono::high_resolution_clock::now();
            blend_ms = std::chrono::duration<double, std::milli>(t_b1 - t_b0).count();
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
            auto t_b0 = std::chrono::high_resolution_clock::now();
            for (int c = 0; c < 3; ++c) oc[c] = bc[c].mul(wBase) + tc[c].mul(wTop);
            cv::Mat outF; cv::merge(oc, outF);
            outF.convertTo(blended, CV_8U, 255.0);
            auto t_b1 = std::chrono::high_resolution_clock::now();
            blend_ms = std::chrono::duration<double, std::milli>(t_b1 - t_b0).count();
        }

        pano = blended;

        // Seam quality on overlap region
        double seam_mean = 0.0, seam_max = 0.0;
        {
            cv::Mat grayA, grayB; cv::cvtColor(canvas, grayA, cv::COLOR_BGR2GRAY); cv::cvtColor(warped, grayB, cv::COLOR_BGR2GRAY);
            cv::Mat overlap; cv::bitwise_and(mask, (mask>0), overlap); // mask itself
            cv::Mat diff;
            cv::absdiff(grayA, grayB, diff);
            cv::Scalar meanVal, stdVal; cv::meanStdDev(diff, meanVal, stdVal, overlap);
            seam_mean = meanVal[0];
            double minv, maxv; cv::minMaxLoc(diff, &minv, &maxv, nullptr, nullptr, overlap);
            seam_max = maxv;
        }

        // Stitch CSV row
        const std::string sHead = "run_id,detector,thresh_px,blending,warp_time_ms,blend_time_ms,seam_error_mean,seam_error_max,out_w,out_h";
        std::snprintf(rowbuf, sizeof(rowbuf), "%s,%s,%.3f,%s,%.3f,%.3f,%.6f,%.6f,%d,%d",
                      run_id.c_str(), toString(detector).c_str(), reprojThresh, toString(blendMode).c_str(),
                      warp_ms, blend_ms, seam_mean, seam_max, pano.cols, pano.rows);
        writeCsvRow(outDir + "/stitch.csv", sHead, rowbuf);
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
