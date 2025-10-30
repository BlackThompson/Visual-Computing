// Core application loop covering capture, CPU/GPU pipelines, and GUI handling.

#include "Application.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <opencv2/imgproc.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <array>
#include <iomanip>
#include <ctime>

namespace
{
    constexpr int kDefaultCamera = 0;
    constexpr int kInitialWidth = 1280;
    constexpr int kInitialHeight = 720;

    constexpr const char* kGlVersion = "#version 330";

    std::string loadTextFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::in);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + path);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    cv::Mat convertBgrToRgba(const cv::Mat& frameBgr)
    {
        cv::Mat rgba;
        cv::cvtColor(frameBgr, rgba, cv::COLOR_BGR2RGBA);
        // OpenCV stores images with the origin at the top-left whereas OpenGL
        // assumes the origin at the bottom-left. Flipping along both axes keeps
        // the orientation consistent between the capture result and the quad.
        cv::flip(rgba, rgba, 0);
        cv::flip(rgba, rgba, 1);
        return rgba;
    }

    std::string shaderPath(const std::string& relative)
    {
        return "shaders/" + relative;
    }

    std::string makeTimestampedFilename(const std::string& prefix,
                                        const std::string& extension)
    {
        const auto now = std::chrono::system_clock::now();
        const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &nowTime);
#else
        localtime_r(&nowTime, &tm);
#endif
        std::ostringstream oss;
        oss << prefix << std::put_time(&tm, "%Y%m%d_%H%M%S") << extension;
        return oss.str();
    }

    constexpr bool kDebugBuild =
#ifdef NDEBUG
        false;
#else
        true;
#endif
}

Application* Application::instance_ = nullptr;

Application::Application()
{
    if (instance_ != nullptr)
    {
        throw std::runtime_error("Only one Application instance is permitted.");
    }
    instance_ = this;
}

Application::~Application()
{
    shutdownImGui();
    shutdownWindow();
    instance_ = nullptr;
}

void Application::run()
{
    initialiseWindow();
    initialiseOpenGL();
    initialiseImGui();
    initialiseCamera(kDefaultCamera);
    loadShaders();
    createQuad();

    lastFrameTime_ = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window_))
    {
        glfwPollEvents();

        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> dt = now - lastFrameTime_;
        lastFrameTime_ = now;

        if (!captureFrame())
        {
            continue;
        }

        updateTransformFromMouse(dt.count());

        double cpuProcessMs = 0.0;
        double gpuProcessMs = 0.0;
        double renderCpuMs = 0.0;

        if (backend_ == ExecutionBackend::CPU)
        {
            const auto cpuStart = std::chrono::steady_clock::now();
            updateCpuPipeline();
            const auto cpuEnd = std::chrono::steady_clock::now();
            cpuProcessMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
        }
        else
        {
            const auto gpuStart = std::chrono::steady_clock::now();
            updateGpuPipeline();
            const auto gpuEnd = std::chrono::steady_clock::now();
            gpuProcessMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        renderGui();

        ImGui::Render();

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        const auto renderStart = std::chrono::steady_clock::now();
        renderFrame();
        const auto renderEnd = std::chrono::steady_clock::now();
        renderCpuMs = std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_);

        updateTimingMetrics(cpuProcessMs, gpuProcessMs, renderCpuMs, dt.count());
        updatePerformance(dt.count(), renderCpuMs, cpuProcessMs, gpuProcessMs, duplicateFrame_);

        if (requestShaderReload_)
        {
            loadShaders();
            requestShaderReload_ = false;
        }
    }

}

void Application::initialiseWindow()
{
    if (glfwInit() == GLFW_FALSE)
    {
        throw std::runtime_error("Failed to initialise GLFW.");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window_ = glfwCreateWindow(kInitialWidth, kInitialHeight,
                               "Visual Computing Lab - Assignment 2", nullptr, nullptr);
    if (!window_)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window.");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
    glfwSetCursorPosCallback(window_, cursorPositionCallback);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
}

void Application::initialiseOpenGL()
{
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        throw std::runtime_error("Failed to initialise GLAD.");
    }
}

void Application::initialiseImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(kGlVersion);
}

void Application::initialiseCamera(int deviceIndex)
{
    if (!camera_.open(deviceIndex, cv::CAP_DSHOW))
    {
        throw std::runtime_error("Unable to open default camera.");
    }

    captureResolution_ = kResolutionOptions_[resolutionIndex_].size;
    setCaptureResolution(captureResolution_);
}

void Application::setCaptureResolution(const cv::Size& size)
{
    captureResolution_ = size;
    if (!camera_.isOpened())
    {
        return;
    }

    camera_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    camera_.set(cv::CAP_PROP_FPS, kTargetFps);
    camera_.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(size.width));
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(size.height));
    camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    camera_.set(cv::CAP_PROP_FPS, kTargetFps);

    for (int i = 0; i < 5; ++i)
    {
        camera_.grab();
    }

    currentFrameBgr_.release();
    cpuProcessedBgr_.release();
    previousFrameGray_.release();
    duplicateFrame_ = false;
}

void Application::shutdownImGui()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Application::shutdownWindow()
{
    if (texture_ != 0)
    {
        glDeleteTextures(1, &texture_);
    }
    if (ebo_ != 0)
    {
        glDeleteBuffers(1, &ebo_);
    }
    if (vbo_ != 0)
    {
        glDeleteBuffers(1, &vbo_);
    }
    if (vao_ != 0)
    {
        glDeleteVertexArrays(1, &vao_);
    }
    if (window_)
    {
        glfwDestroyWindow(window_);
        glfwTerminate();
        window_ = nullptr;
    }
}

void Application::loadShaders()
{
    const std::string vertexSrc = loadTextFile(shaderPath("textured_quad.vert"));
    const std::string passthroughSrc = loadTextFile(shaderPath("pass_through.frag"));
    const std::string pixelateSrc = loadTextFile(shaderPath("pixelate.frag"));
    const std::string comicSrc = loadTextFile(shaderPath("comic.frag"));
    const std::string edgeSrc = loadTextFile(shaderPath("edge.frag"));

    passThroughProgram_ = ShaderProgram(vertexSrc, passthroughSrc);
    pixelateProgram_ = ShaderProgram(vertexSrc, pixelateSrc);
    comicProgram_ = ShaderProgram(vertexSrc, comicSrc);
    edgeProgram_ = ShaderProgram(vertexSrc, edgeSrc);

    switchFilterShader();
}

void Application::createQuad()
{
    const std::array<float, 20> vertices = {
        // positions      // tex coords
        -1.0f, -1.0f,     0.0f, 0.0f,
         1.0f, -1.0f,     1.0f, 0.0f,
         1.0f,  1.0f,     1.0f, 1.0f,
        -1.0f,  1.0f,     0.0f, 1.0f,
        // padded to align (OpenGL requires stride spec)
    };

    const std::array<unsigned int, 6> indices = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenBuffers(1, &ebo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices.data(), GL_STATIC_DRAW);

    const GLsizei stride = 4 * sizeof(float);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(2 * sizeof(float)));

    glBindVertexArray(0);
}

void Application::createTexture(int width, int height)
{
    if (texture_ == 0)
    {
        glGenTextures(1, &texture_);
    }
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool Application::captureFrame()
{
    if (!camera_.isOpened())
    {
        return false;
    }

    camera_ >> currentFrameBgr_;
    if (currentFrameBgr_.empty())
    {
        return false;
    }

    cv::Mat currentGray;
    cv::cvtColor(currentFrameBgr_, currentGray, cv::COLOR_BGR2GRAY);
    if (!previousFrameGray_.empty() &&
        previousFrameGray_.size() == currentGray.size())
    {
        cv::Mat diff;
        cv::absdiff(currentGray, previousFrameGray_, diff);
        duplicateFrame_ = cv::mean(diff)[0] < kDuplicateThreshold;
    }
    else
    {
        duplicateFrame_ = false;
    }
    currentGray.copyTo(previousFrameGray_);

    createTexture(currentFrameBgr_.cols, currentFrameBgr_.rows);
    return true;
}

void Application::updateCpuPipeline()
{
    cpuProcessedBgr_ = FrameProcessor::applyFilter(
        currentFrameBgr_, currentFilter_, filterParams_);

    if (transformEnabled_)
    {
        FrameProcessor::applyTransform(cpuProcessedBgr_, transform_);
    }

    uploadFrameToTexture(cpuProcessedBgr_);
}

void Application::updateGpuPipeline()
{
    // GPU filters operate directly on the raw input frame.
    uploadFrameToTexture(currentFrameBgr_);
}

void Application::uploadFrameToTexture(const cv::Mat& frameBgr)
{
    cv::Mat rgba = convertBgrToRgba(frameBgr);

    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    rgba.cols, rgba.rows,
                    GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Application::renderFrame()
{
    if (currentFrameBgr_.empty() || texture_ == 0)
    {
        return;
    }

    switchFilterShader();
    activeProgram_->use();

    const GLint texLocation = glGetUniformLocation(activeProgram_->id(), "uFrame");
    if (texLocation >= 0)
    {
        glUniform1i(texLocation, 0);
    }

    const GLint texSizeLocation = glGetUniformLocation(activeProgram_->id(), "uTextureSize");
    if (texSizeLocation >= 0)
    {
        glUniform2f(texSizeLocation,
                    static_cast<float>(currentFrameBgr_.cols),
                    static_cast<float>(currentFrameBgr_.rows));
    }

    const bool isCpu = backend_ == ExecutionBackend::CPU;
    const bool transformActive = transformEnabled_ && transform_.isActive();
    const GLint transformEnabledLoc = glGetUniformLocation(activeProgram_->id(), "uTransformEnabled");
    if (transformEnabledLoc >= 0)
    {
        glUniform1i(transformEnabledLoc, (isCpu ? 0 : (transformActive ? 1 : 0)));
    }

    if (!isCpu)
    {
        const GLint transformLoc = glGetUniformLocation(activeProgram_->id(), "uTexTransform");
        if (transformLoc >= 0)
        {
            std::array<float, 9> matrix = {
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 1.0f
            };

            if (transformActive && !currentFrameBgr_.empty())
            {
                cv::Mat affine = FrameProcessor::computeAffineMatrix(
                    currentFrameBgr_.size(), transform_);

                cv::Mat affine3x3 = cv::Mat::eye(3, 3, CV_32F);
                affine.copyTo(affine3x3(cv::Rect(0, 0, 3, 2)));

                cv::Mat flip = cv::Mat::eye(3, 3, CV_32F);
                flip.at<float>(0, 0) = -1.0f;
                flip.at<float>(1, 1) = -1.0f;
                flip.at<float>(0, 2) = static_cast<float>(currentFrameBgr_.cols);
                flip.at<float>(1, 2) = static_cast<float>(currentFrameBgr_.rows);

                cv::Mat adjusted = flip * affine3x3 * flip;

                matrix = {
                    adjusted.at<float>(0, 0), adjusted.at<float>(1, 0), 0.0f,
                    adjusted.at<float>(0, 1), adjusted.at<float>(1, 1), 0.0f,
                    adjusted.at<float>(0, 2), adjusted.at<float>(1, 2), 1.0f
                };
            }

            glUniformMatrix3fv(transformLoc, 1, GL_FALSE, matrix.data());
        }
    }

    const GLint pixelationLocation = glGetUniformLocation(activeProgram_->id(), "uPixelBlockSize");
    if (pixelationLocation >= 0)
    {
        glUniform1f(pixelationLocation, static_cast<float>(filterParams_.pixelate.blockSize));
    }

    const GLint colorLevelsLoc = glGetUniformLocation(activeProgram_->id(), "uColorLevels");
    if (colorLevelsLoc >= 0)
    {
        glUniform1i(colorLevelsLoc, std::max(2, filterParams_.comic.colorLevels));
    }
    const GLint edgeThresholdLoc = glGetUniformLocation(activeProgram_->id(), "uEdgeThreshold");
    if (edgeThresholdLoc >= 0)
    {
        glUniform1f(edgeThresholdLoc, filterParams_.comic.edgeThreshold);
    }
    const GLint sobelThresholdLoc = glGetUniformLocation(activeProgram_->id(), "uEdgeFilterThreshold");
    if (sobelThresholdLoc >= 0)
    {
        glUniform1f(sobelThresholdLoc, filterParams_.edge.threshold);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);

    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void Application::renderGui()
{
    ImGui::SetNextWindowSize(ImVec2(620.0f, 720.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Controls");

    if (ImGui::BeginCombo("Resolution", kResolutionOptions_[resolutionIndex_].label))
    {
        for (int i = 0; i < static_cast<int>(kResolutionOptions_.size()); ++i)
        {
            const bool selected = (i == resolutionIndex_);
            if (ImGui::Selectable(kResolutionOptions_[i].label, selected))
            {
                resolutionIndex_ = i;
                setCaptureResolution(kResolutionOptions_[i].size);
            }
            if (selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    if (!currentFrameBgr_.empty())
    {
        ImGui::Text("Active Frame: %dx%d", currentFrameBgr_.cols, currentFrameBgr_.rows);
    }

    ImGui::Separator();

    const char* backendLabels[] = { "CPU", "GPU" };
    int backendIndex = backend_ == ExecutionBackend::CPU ? 0 : 1;
    if (ImGui::Combo("Backend", &backendIndex, backendLabels, IM_ARRAYSIZE(backendLabels)))
    {
        backend_ = (backendIndex == 0) ? ExecutionBackend::CPU : ExecutionBackend::GPU;
    }

    const char* filterLabels[] = { "None", "Pixelate", "Comic", "Edge" };
    int filterIndex = static_cast<int>(currentFilter_);
    if (ImGui::Combo("Filter", &filterIndex, filterLabels, IM_ARRAYSIZE(filterLabels)))
    {
        currentFilter_ = static_cast<FilterType>(filterIndex);
    }

    if (currentFilter_ == FilterType::Pixelate)
    {
        ImGui::SliderInt("Block Size", &filterParams_.pixelate.blockSize, 1, 64);
    }
    else if (currentFilter_ == FilterType::Comic)
    {
        ImGui::SliderInt("Colour Levels", &filterParams_.comic.colorLevels, 2, 8);
        ImGui::SliderFloat("Edge Threshold", &filterParams_.comic.edgeThreshold, 0.05f, 0.75f);
    }
    else if (currentFilter_ == FilterType::Edge)
    {
        ImGui::SliderFloat("Edge Threshold", &filterParams_.edge.threshold, 0.05f, 1.0f);
    }

    ImGui::Separator();
    ImGui::Checkbox("Enable Transform", &transformEnabled_);
    ImGui::SliderFloat("Translate X", &transform_.translateX, -200.0f, 200.0f);
    ImGui::SliderFloat("Translate Y", &transform_.translateY, -200.0f, 200.0f);
    ImGui::SliderFloat("Rotation", &transform_.rotationDegrees, -180.0f, 180.0f);
    ImGui::SliderFloat("Scale", &transform_.scale, 0.2f, 3.0f);

    if (ImGui::Button("Reset Transform"))
    {
        transform_ = TransformParams{};
    }

    if (ImGui::Button("Reload Shaders"))
    {
        requestShaderReload_ = true;
    }

    ImGui::Separator();
    ImGui::Text("Frame Time: %.2f ms (%.1f FPS)", lastFrameTimeMs_, lastFrameTimeMs_ > 0.0 ? 1000.0 / lastFrameTimeMs_ : 0.0);
    ImGui::Text("CPU Processing: %.2f ms", lastCpuProcessMs_);
    ImGui::Text("GPU Upload: %.2f ms", lastGpuProcessMs_);
    ImGui::Text("Render Submission: %.2f ms", lastRenderCpuMs_);

    ImGui::Separator();
    if (ImGui::BeginTable("PerformanceTable", 8,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Configuration", ImGuiTableColumnFlags_WidthStretch, 0.35f);
        ImGui::TableSetupColumn("Avg FPS", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Frame (ms)", ImGuiTableColumnFlags_WidthFixed, 95.0f);
        ImGui::TableSetupColumn("CPU (ms)", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("GPU Upload (ms)", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("GPU Submit (ms)", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("End-to-End (ms)", ImGuiTableColumnFlags_WidthFixed, 130.0f);
        ImGui::TableSetupColumn("Dup Rate (%)", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableHeadersRow();

        for (const auto& [key, stats] : performance_.data())
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(performance_.describeKey(key).c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.2f", stats.averageFps());
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.2f", stats.averageFrameMs());
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.2f", stats.averageCpuMs());
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.2f", stats.averageGpuUploadMs());
            ImGui::TableSetColumnIndex(5);
            ImGui::Text("%.2f", stats.averageRenderMs());
            ImGui::TableSetColumnIndex(6);
            ImGui::Text("%.2f", stats.averageEndToEndMs());
            ImGui::TableSetColumnIndex(7);
            ImGui::Text("%.2f", stats.duplicationRatePct());
        }

        ImGui::EndTable();
    }

    if (ImGui::Button("Export CSV"))
    {
        exportPerformanceCsv();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear Metrics"))
    {
        performance_.clear();
        lastExportMessage_.clear();
        activePerfKey_.reset();
    }
    if (!lastExportMessage_.empty())
    {
        ImGui::TextWrapped("%s", lastExportMessage_.c_str());
    }

    ImGui::End();
}

void Application::updateTransformFromMouse(double /*deltaTime*/)
{
    if (!transformEnabled_)
    {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        return;
    }

    if (mouseDragging_)
    {
        double mouseX, mouseY;
        glfwGetCursorPos(window_, &mouseX, &mouseY);
        const double dx = mouseX - lastMouseX_;
        const double dy = mouseY - lastMouseY_;
        transform_.translateX += static_cast<float>(dx);
        transform_.translateY += static_cast<float>(dy);
        lastMouseX_ = mouseX;
        lastMouseY_ = mouseY;
    }
}

void Application::updatePerformance(double frameTimeMs,
                                    double renderTimeMs,
                                    double cpuTimeMs,
                                    double gpuUploadMs,
                                    bool duplicateFrame)
{
    PerformanceKey key;
    key.backend = backend_;
    key.filter = currentFilter_;
    key.resolution = { currentFrameBgr_.cols, currentFrameBgr_.rows };
    key.debugBuild = kDebugBuild;
    key.transformationEnabled = transformEnabled_ && transform_.isActive();

    if (!activePerfKey_.has_value() || !(*activePerfKey_ == key))
    {
        performance_.resetSamples(key);
        activePerfKey_ = key;
    }

    performance_.pushSample(key, frameTimeMs, renderTimeMs, cpuTimeMs, gpuUploadMs, duplicateFrame);
}

void Application::updateTimingMetrics(double cpuMs, double gpuMs, double renderMs, double frameMs)
{
    lastCpuProcessMs_ = cpuMs;
    lastGpuProcessMs_ = gpuMs;
    lastRenderCpuMs_ = renderMs;
    lastFrameTimeMs_ = frameMs;
}

void Application::exportPerformanceCsv()
{
    const auto& dataset = performance_.data();
    if (dataset.empty())
    {
        lastExportMessage_ = "No performance samples available to export.";
        return;
    }

    const std::string filename = makeTimestampedFilename("performance_", ".csv");
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if (!file.is_open())
    {
        lastExportMessage_ = "Failed to create " + filename;
        return;
    }

    file << "Configuration,Average FPS,Frame Time (ms),CPU Time (ms),GPU Upload (ms),"
            "GPU Submit (ms),End-to-End (ms),Duplication Rate (%),Sample Count\n";
    file << std::fixed << std::setprecision(2);

    for (const auto& [key, stats] : dataset)
    {
        file << '"' << performance_.describeKey(key) << '"' << ','
             << stats.averageFps() << ','
             << stats.averageFrameMs() << ','
             << stats.averageCpuMs() << ','
             << stats.averageGpuUploadMs() << ','
             << stats.averageRenderMs() << ','
             << stats.averageEndToEndMs() << ','
             << stats.duplicationRatePct() << ','
             << stats.frameTimesMs.size()
             << '\n';
    }

    lastExportMessage_ = "Exported performance metrics to " + filename;
}

void Application::switchFilterShader()
{
    if (backend_ == ExecutionBackend::CPU)
    {
        activeProgram_ = &passThroughProgram_;
        return;
    }

    switch (currentFilter_)
    {
        case FilterType::Pixelate:
            activeProgram_ = &pixelateProgram_;
            break;
        case FilterType::Comic:
            activeProgram_ = &comicProgram_;
            break;
        case FilterType::Edge:
            activeProgram_ = &edgeProgram_;
            break;
        case FilterType::None:
        default:
            activeProgram_ = &passThroughProgram_;
            break;
    }
}

void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void Application::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (instance_ == nullptr) return;
    if (instance_->mouseDragging_)
    {
        instance_->lastMouseX_ = xpos;
        instance_->lastMouseY_ = ypos;
    }
}

void Application::mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    if (instance_ == nullptr) return;

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            instance_->mouseDragging_ = true;
            glfwGetCursorPos(window, &instance_->lastMouseX_, &instance_->lastMouseY_);
        }
        else if (action == GLFW_RELEASE)
        {
            instance_->mouseDragging_ = false;
        }
    }
}
