# Real-Time Video Processing with Dear ImGui, OpenGL, and OpenCV

This project implements a Windows-first, real-time webcam processing pipeline combining OpenCV (for capture and CPU processing), OpenGL/GLSL (for rendering and GPU filters), and Dear ImGui (for the interactive GUI). The application was developed for the Visual Computing course assignment on advanced shader-based effects.

## Features
- Live webcam capture routed into an OpenGL texture.
- CPU and GPU implementations for three real-time filters: **Pixelate**, **Comic Book**, and **Edge Highlight**.
- Optional affine transform (translation, rotation, and scaling) controllable via GUI sliders or direct mouse dragging in the viewport.
- Dear ImGui overlay for switching filters, toggling CPU/GPU backends, tuning filter parameters, and selecting capture resolutions (1280x720, 960x540, 640x360) at runtime.
- Built-in performance logger that tracks average FPS for every backend/filter/resolution/transform combination encountered during a session, plus live processing-time readouts.

## Directory Layout
```
.
|-- CMakeLists.txt          # Build configuration (FetchContent for glfw/glad/imgui)
|-- include/                # Project headers
|-- src/                    # C++ sources (application, CPU filters)
|-- shaders/                # GLSL vertex/fragment shaders (GPU filters)
|-- docs/Report.md          # Detailed methodology and results write-up
`-- README.md               # This file
```

## Build Instructions (Windows/MSVC)
1. **Install prerequisites**
   - [CMake >= 3.20](https://cmake.org/download/)
   - Visual Studio 2022 with the *Desktop development with C++* workload.
   - [OpenCV](https://opencv.org/releases/) 4.x built for MSVC (set `OpenCV_DIR` to its CMake package directory).
2. **Configure the project**
   ```powershell
   mkdir build
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:/path/to/opencv/build"
   ```
   CMake downloads glfw, glad, and Dear ImGui via `FetchContent`. Ensure outbound HTTPS traffic is allowed during the first configure step.
3. **Build**
   ```powershell
   cmake --build build --config Release
   ```
4. **Run**
   ```powershell
   .\build\Release\VisualComputingPipeline.exe
   ```
   Visual Studio users can launch the `VisualComputingPipeline` target directly. The debugger working directory is set to the repository root so shader lookups succeed without manual copies.

## Clean Rebuild Workflow
When dependencies change or the CMake cache becomes stale, perform a clean rebuild and stage the required runtime DLLs alongside the executable:
1. **Remove the previous build tree**
   ```powershell
   if (Test-Path build) { Remove-Item -Recurse -Force build }
   ```
2. **Configure and regenerate project files**
   ```powershell
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:/path/to/opencv/build"
   ```
3. **Compile the Release configuration**
   ```powershell
   cmake --build build --config Release
   ```
4. **Copy OpenCV runtime libraries next to the executable**
   ```powershell
   Copy-Item "C:/path/to/opencv/build/x64/vc16/bin/opencv_world4*.dll" -Destination build/Release
   Copy-Item "C:/path/to/opencv/build/x64/vc16/bin/opencv_videoio_ffmpeg4*_64.dll" -Destination build/Release
   ```
   Adjust the filenames to match your OpenCV version. Place any additional OpenCV DLLs required by your installation in the same `build/Release` directory so `VisualComputingPipeline.exe` can resolve them at runtime.
5. **Launch**
   ```powershell
   .\build\Release\VisualComputingPipeline.exe
   ```

## Usage Guide
- The viewport shows the live camera feed.
- Use the **Resolution** combo box to switch the capture device between 1280x720, 960x540, and 640x360. The active frame size is displayed beside the selector.
- Use the **Backend** combo box to switch between CPU and GPU processing.
- Use the **Filter** combo box to choose `None`, `Pixelate`, `Comic`, or `Edge`.
- Each filter reveals specific controls:
  - *Pixelate*: Block size in pixels.
  - *Comic*: Colour quantisation level and edge threshold.
  - *Edge*: Sobel gradient threshold controlling edge visibility.
- Tick **Enable Transform** to activate the affine manipulation controls:
  - GUI sliders adjust translation (+/-200 px), rotation (+/-180 deg), and uniform scale (0.2-3x).
  - Click and drag inside the viewport to translate the image interactively; CPU and GPU back-ends now share identical transform math.
  - Press **Reset Transform** to restore the identity transform.
- Live timing labels show per-frame CPU processing, GPU upload, render submission, and frame time measurements.
- The **Performance** table accumulates running averages for FPS, frame time, CPU time, GPU upload, GPU submit, end-to-end latency, and frame-duplication rate per backend/filter/resolution/transform combination. Each switch resets the window so readings stabilise after a few frames.
- **Reload Shaders** recompiles the GLSL files from `shaders/`, enabling rapid iteration on shader logic without restarting the executable.
- Press **Export CSV** to dump the current performance metrics to a timestamped CSV in the working directory.
- Press **Clear Metrics** to reset the in-app statistics before starting a new experiment.

## Implementation Notes
- **Initialization Path**  
  `Application::run()` wires together GLFW window creation, GLAD loader setup, Dear ImGui initialisation, shader compilation, and OpenCV camera configuration. Unified error handling ensures clean shutdowns.
- **Frame Ingestion and CPU Processing**  
  `Application::captureFrame()` grabs frames in BGR format from OpenCV. `FrameProcessor::applyFilter()` runs the selected CPU filter (pixelation, comic stylisation, or Sobel edge highlight), while `FrameProcessor::applyTransform()` applies the affine warp using `cv::warpAffine`.
- **Camera Capture Format**  
  The camera is forced to MJPEG (`CAP_PROP_FOURCC = MJPG`) and 30 FPS for each resolution change to ensure stable frame rates across 1280x720, 960x540, and 640x360 streams.
- **Frame Duplication Detection**  
  Consecutive frames are compared in grayscale; a low per-pixel absolute difference flags driver-level duplication and populates the duplication rate column.
- **CSV Export**  
  A timestamped `performance_YYYYMMDD_HHMMSS.csv` is written via the GUI, capturing the same metrics shown in the table for offline analysis.
- **Metrics Reset**  
  The GUI ``Clear Metrics`` button wipes rolling statistics so a fresh test run starts without residual averages.
- **Texture Upload**  
  `Application::uploadFrameToTexture()` converts BGR to RGBA and updates a persistent OpenGL texture with `glTexSubImage2D`, minimising reallocations.
- **GPU Pipeline**  
  The vertex shader (`textured_quad.vert`) renders a full-screen quad. Fragment shaders implement `pass_through`, `pixelate`, `comic`, and `edge` effects. Uniforms mirror the OpenCV affine matrix so CPU and GPU transforms stay pixel-identical, and out-of-bounds samples return black to match CPU `cv::warpAffine` behaviour.
- **Main Loop Structure**  
  Every frame: poll events -> capture frame -> update CPU/GPU pipeline -> build Dear ImGui UI -> render quad -> submit ImGui draw data -> swap buffers -> record performance metrics.

## Troubleshooting
- *Camera not found*: Verify another application is not locking the webcam. Update the `CAP_DSHOW` backend index if you have multiple cameras.
- *Missing OpenCV*: Ensure `OpenCV_DIR` points to the directory containing `OpenCVConfig.cmake`.
- *Linker errors (glfw/glad/imgui)*: Delete the `build/` folder and reconfigure to force a clean FetchContent download.
- *Black screen*: Check the console for shader compilation errors; use **Reload Shaders** after editing GLSL files.

## License
Coursework deliverable. External dependencies (glfw, glad, Dear ImGui, OpenCV) retain their respective licenses.









