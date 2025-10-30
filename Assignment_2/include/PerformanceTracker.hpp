// Tracks frame timing statistics for different filter/backend configurations.
// The results feed both the on-screen Dear ImGui table and the written report.

#pragma once

#include "Types.hpp"

#include <chrono>
#include <deque>
#include <map>
#include <string>
#include <cstdint>

struct PerformanceStats
{
    static constexpr std::size_t kMaxSamples = 90;

    std::deque<double> frameTimesMs;
    std::deque<double> renderTimesMs;
    std::deque<double> cpuTimesMs;
    std::deque<double> gpuUploadTimesMs;
    std::deque<int> duplicateFlags;
    double sumFrameMs = 0.0;
    double sumRenderMs = 0.0;
    double sumCpuMs = 0.0;
    double sumGpuUploadMs = 0.0;
    int sumDuplicateFlags = 0;
    std::uint64_t totalSamples = 0;
    bool primed = false;

    void addSample(double frameTimeMs,
                   double renderTimeMs,
                   double cpuTimeMs,
                   double gpuUploadMs,
                   bool duplicateFrame)
    {
        if (!primed)
        {
            primed = true;
            return;
        }

        frameTimesMs.push_back(frameTimeMs);
        sumFrameMs += frameTimeMs;
        if (frameTimesMs.size() > kMaxSamples)
        {
            sumFrameMs -= frameTimesMs.front();
            frameTimesMs.pop_front();
        }

        renderTimesMs.push_back(renderTimeMs);
        sumRenderMs += renderTimeMs;
        if (renderTimesMs.size() > kMaxSamples)
        {
            sumRenderMs -= renderTimesMs.front();
            renderTimesMs.pop_front();
        }

        cpuTimesMs.push_back(cpuTimeMs);
        sumCpuMs += cpuTimeMs;
        if (cpuTimesMs.size() > kMaxSamples)
        {
            sumCpuMs -= cpuTimesMs.front();
            cpuTimesMs.pop_front();
        }

        gpuUploadTimesMs.push_back(gpuUploadMs);
        sumGpuUploadMs += gpuUploadMs;
        if (gpuUploadTimesMs.size() > kMaxSamples)
        {
            sumGpuUploadMs -= gpuUploadTimesMs.front();
            gpuUploadTimesMs.pop_front();
        }

        const int duplicateInt = duplicateFrame ? 1 : 0;
        duplicateFlags.push_back(duplicateInt);
        sumDuplicateFlags += duplicateInt;
        if (duplicateFlags.size() > kMaxSamples)
        {
            sumDuplicateFlags -= duplicateFlags.front();
            duplicateFlags.pop_front();
        }

        ++totalSamples;
    }

    [[nodiscard]] double averageFps() const
    {
        if (frameTimesMs.empty() || sumFrameMs <= 0.0)
        {
            return 0.0;
        }
        double avgFrameMs = sumFrameMs / static_cast<double>(frameTimesMs.size());
        return avgFrameMs > 0.0 ? 1000.0 / avgFrameMs : 0.0;
    }

    [[nodiscard]] double averageRenderMs() const
    {
        if (renderTimesMs.empty())
        {
            return 0.0;
        }
        return sumRenderMs / static_cast<double>(renderTimesMs.size());
    }

    [[nodiscard]] double averageFrameMs() const
    {
        if (frameTimesMs.empty())
        {
            return 0.0;
        }
        return sumFrameMs / static_cast<double>(frameTimesMs.size());
    }

    [[nodiscard]] double averageCpuMs() const
    {
        if (cpuTimesMs.empty())
        {
            return 0.0;
        }
        return sumCpuMs / static_cast<double>(cpuTimesMs.size());
    }

    [[nodiscard]] double averageGpuUploadMs() const
    {
        if (gpuUploadTimesMs.empty())
        {
            return 0.0;
        }
        return sumGpuUploadMs / static_cast<double>(gpuUploadTimesMs.size());
    }

    [[nodiscard]] double averageEndToEndMs() const
    {
        return averageCpuMs() + averageGpuUploadMs() + averageRenderMs();
    }

    [[nodiscard]] double duplicationRatePct() const
    {
        if (duplicateFlags.empty())
        {
            return 0.0;
        }
        return static_cast<double>(sumDuplicateFlags) * 100.0 /
               static_cast<double>(duplicateFlags.size());
    }
};

class PerformanceTracker
{
public:
    void pushSample(const PerformanceKey& key,
                    double frameTimeMs,
                    double renderTimeMs,
                    double cpuTimeMs,
                    double gpuUploadMs,
                    bool duplicateFrame)
    {
        auto& stats = dataset_[key];
        stats.addSample(frameTimeMs, renderTimeMs, cpuTimeMs, gpuUploadMs, duplicateFrame);
    }

    void resetSamples(const PerformanceKey& key)
    {
        auto it = dataset_.find(key);
        if (it != dataset_.end())
        {
            it->second.frameTimesMs.clear();
            it->second.renderTimesMs.clear();
            it->second.cpuTimesMs.clear();
            it->second.gpuUploadTimesMs.clear();
            it->second.duplicateFlags.clear();
            it->second.sumFrameMs = 0.0;
            it->second.sumRenderMs = 0.0;
            it->second.sumCpuMs = 0.0;
            it->second.sumGpuUploadMs = 0.0;
            it->second.sumDuplicateFlags = 0;
            it->second.totalSamples = 0;
            it->second.primed = false;
        }
    }

    [[nodiscard]] const std::map<PerformanceKey, PerformanceStats>& data() const
    {
        return dataset_;
    }

    [[nodiscard]] std::string describeKey(const PerformanceKey& key) const
    {
        const auto& res = key.resolution;
        return toString(key.filter) + " | " +
               toString(key.backend) + " | " +
               std::to_string(res.first) + "x" + std::to_string(res.second) + " | " +
               (key.debugBuild ? "Debug" : "Release") + " | " +
               (key.transformationEnabled ? "Transform ON" : "Transform OFF");
    }

    void clear()
    {
        dataset_.clear();
    }

private:
    std::map<PerformanceKey, PerformanceStats> dataset_;
};
