#include <opencv2/opencv.hpp>
#include <iostream>
#include "preprocess.hpp"
#include "features.hpp"
#include "matching.hpp"
#include "homography.hpp"
#include "warp.hpp"
#include "blend.hpp"
#include "stitch.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: panorama <img1> <img2> [img3 ...]\n";
        std::cout << "Options: --det [sift|orb|akaze] --blend [overlay|feather] --ratio <0.5-0.95> --ransac <iters> --th <px> --debug\n";
        return 0;
    }
    vc::Detector det = vc::Detector::ORB;
    vc::BlendMode bm = vc::BlendMode::FEATHER;
    double ratio = 0.75;
    int ransacIter = 1000;
    double reproj = 3.0;
    bool debug = false;

    std::vector<std::string> paths;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--det" && i+1 < argc) {
            std::string v = argv[++i];
            if (v == "sift") det = vc::Detector::SIFT;
            else if (v == "orb") det = vc::Detector::ORB;
            else if (v == "akaze") det = vc::Detector::AKAZE;
        } else if (a == "--blend" && i+1 < argc) {
            std::string v = argv[++i];
            if (v == "overlay") bm = vc::BlendMode::OVERLAY;
            else if (v == "feather") bm = vc::BlendMode::FEATHER;
        } else if (a == "--ratio" && i+1 < argc) {
            ratio = std::stod(argv[++i]);
        } else if (a == "--ransac" && i+1 < argc) {
            ransacIter = std::stoi(argv[++i]);
        } else if (a == "--th" && i+1 < argc) {
            reproj = std::stod(argv[++i]);
        } else if (a == "--debug") {
            debug = true;
        } else if (!a.empty() && a[0] != '-') {
            paths.push_back(a);
        }
    }

    std::vector<cv::Mat> imgs;
    for (const auto& p : paths) {
        cv::Mat img = cv::imread(p);
        if (img.empty()) { std::cerr << "Failed to read " << p << std::endl; return 1; }
        imgs.push_back(img);
    }
    if (imgs.size() < 2) { std::cout << "Need >=2 images\n"; return 0; }
    // Create unique run directory: results/run_YYYYmmdd_HHMMSS
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "results/run_%04d%02d%02d_%02d%02d%02d",
                  tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    std::string outDir = buf;

    cv::Mat pano = vc::stitchImages(imgs, det, bm, ransacIter, reproj, ratio, debug, outDir);
    if (pano.empty()) { std::cerr << "Stitch failed\n"; return 1; }
    std::string outPano = outDir + "/panorama.jpg";
    cv::imwrite(outPano, pano);
    std::cout << "Saved: " << outPano << std::endl;
    return 0;
}
