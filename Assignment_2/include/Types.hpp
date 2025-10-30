// Copyright (c) 2025.
// This header declares shared data structures and enumerations for the
// real-time video processing application.

#pragma once

#include <string>
#include <utility>

// Enumeration describing the available visual filters.
enum class FilterType
{
    None = 0,
    Pixelate,
    Comic,
    Edge
};

// Enumeration describing whether the CPU or GPU path is active.
enum class ExecutionBackend
{
    CPU = 0,
    GPU
};

// Tracks the user-configurable affine transformation parameters.
struct TransformParams
{
    // Translation in pixels relative to the video frame.
    float translateX = 0.0f;
    float translateY = 0.0f;

    // Rotation around the frame centre in degrees.
    float rotationDegrees = 0.0f;

    // Uniform scaling factor applied around the frame centre.
    float scale = 1.0f;

    // Helper signalling whether the transform deviates from the identity.
    [[nodiscard]] bool isActive() const
    {
        return translateX != 0.0f || translateY != 0.0f ||
               rotationDegrees != 0.0f || scale != 1.0f;
    }
};

// Parameter bundles for individual filters. Keeping the values together makes
// it easy to expose them through Dear ImGui widgets while dispatching the same
// configuration to CPU and GPU back-ends.
struct PixelateParams
{
    int blockSize = 8;     // Size of the pixel block in screen pixels.
};

struct ComicParams
{
    int colorLevels = 4;   // Number of discrete colour bands.
    float edgeThreshold = 0.25f;  // Normalised threshold for edge detection.
};

struct EdgeParams
{
    float threshold = 0.2f;    // Normalised gradient threshold for edge visibility.
};

// Convenience structure grouping all tunable parameters under a single handle.
struct FilterParameters
{
    PixelateParams pixelate;
    ComicParams comic;
    EdgeParams edge;
};

// Strongly typed key for performance statistics. The struct doubles as a map
// key thanks to the custom < operator.
struct PerformanceKey
{
    FilterType filter = FilterType::None;
    ExecutionBackend backend = ExecutionBackend::CPU;
    std::pair<int, int> resolution{0, 0};  // width, height
    bool debugBuild = false;
    bool transformationEnabled = false;

    bool operator<(const PerformanceKey& other) const
    {
        const auto filterValue = static_cast<int>(filter);
        const auto otherFilterValue = static_cast<int>(other.filter);
        if (filterValue != otherFilterValue) return filterValue < otherFilterValue;

        const auto backendValue = static_cast<int>(backend);
        const auto otherBackendValue = static_cast<int>(other.backend);
        if (backendValue != otherBackendValue) return backendValue < otherBackendValue;

        if (resolution != other.resolution) return resolution < other.resolution;
        if (debugBuild != other.debugBuild) return debugBuild < other.debugBuild;
        return transformationEnabled < other.transformationEnabled;
    }

    bool operator==(const PerformanceKey& other) const
    {
        return filter == other.filter &&
               backend == other.backend &&
               resolution == other.resolution &&
               debugBuild == other.debugBuild &&
               transformationEnabled == other.transformationEnabled;
    }
};

// Utility for translating enum values into human-friendly labels when
// populating Dear ImGui widgets or report tables.
inline std::string toString(FilterType type)
{
    switch (type)
    {
        case FilterType::None: return "None";
        case FilterType::Pixelate: return "Pixelate";
        case FilterType::Comic: return "Comic";
        case FilterType::Edge: return "Edge";
    }
    return "Unknown";
}

inline std::string toString(ExecutionBackend backend)
{
    switch (backend)
    {
        case ExecutionBackend::CPU: return "CPU";
        case ExecutionBackend::GPU: return "GPU";
    }
    return "Unknown";
}
