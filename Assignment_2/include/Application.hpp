// Orchestrates window creation, ImGui setup, OpenCV capture, and rendering.

#pragma once

#include "FrameProcessor.hpp"
#include "PerformanceTracker.hpp"
#include "ShaderProgram.hpp"
#include "Types.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <optional>
#include <chrono>
#include <string>

class Application
{
public:
    Application();
    ~Application();

    void run();

private:
    void initialiseWindow();
    void initialiseOpenGL();
    void initialiseImGui();
    void initialiseCamera(int deviceIndex);
    void setCaptureResolution(const cv::Size& size);

    void shutdownImGui();
    void shutdownWindow();

    void loadShaders();
    void createQuad();
    void createTexture(int width, int height);

    bool captureFrame();
    void uploadFrameToTexture(const cv::Mat& frameBgr);
    void renderFrame();
    void renderGui();

    void updateCpuPipeline();
    void updateGpuPipeline();

    void updateTransformFromMouse(double deltaTime);
    void updatePerformance(double frameTimeMs,
                           double renderTimeMs,
                           double cpuTimeMs,
                           double gpuUploadMs,
                           bool duplicateFrame);
    void updateTimingMetrics(double cpuMs, double gpuMs, double renderMs, double frameMs);
    void exportPerformanceCsv();

    void switchFilterShader();

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

private:
    GLFWwindow* window_ = nullptr;
    cv::VideoCapture camera_;

    cv::Mat currentFrameBgr_;
    cv::Mat cpuProcessedBgr_;
    cv::Mat previousFrameGray_;

    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint ebo_ = 0;
    GLuint texture_ = 0;

    ShaderProgram passThroughProgram_;
    ShaderProgram pixelateProgram_;
    ShaderProgram comicProgram_;
    ShaderProgram edgeProgram_;
    ShaderProgram* activeProgram_ = nullptr;

    FilterType currentFilter_ = FilterType::None;
    ExecutionBackend backend_ = ExecutionBackend::CPU;
    TransformParams transform_;
    FilterParameters filterParams_;

    PerformanceTracker performance_;
    std::optional<PerformanceKey> activePerfKey_;

    bool transformEnabled_ = true;
    bool requestShaderReload_ = false;

    bool mouseDragging_ = false;
    double lastMouseX_ = 0.0;
    double lastMouseY_ = 0.0;

    std::chrono::steady_clock::time_point lastFrameTime_;

    static Application* instance_;

    cv::Size captureResolution_{1280, 720};
    int resolutionIndex_ = 0;

    double lastCpuProcessMs_ = 0.0;
    double lastGpuProcessMs_ = 0.0;
    double lastRenderCpuMs_ = 0.0;
    double lastFrameTimeMs_ = 0.0;
    bool duplicateFrame_ = false;
    std::string lastExportMessage_;

    struct ResolutionOption
    {
        const char* label;
        cv::Size size;
    };

    inline static const std::array<ResolutionOption, 3> kResolutionOptions_ = {{
        { "1280 x 720", {1280, 720} },
        { "960 x 540", {960, 540} },
        { "640 x 360", {640, 360} }
    }};

    static constexpr double kDuplicateThreshold = 1.5;
    static constexpr double kTargetFps = 30.0;
};
