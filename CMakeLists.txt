cmake_minimum_required(VERSION 3.19)
project(MyOpenCVProject)

set(CMAKE_CXX_STANDARD 17)

# 1. Manually set the library directory for the linker
link_directories("/opt/homebrew/lib")

# 2. Replicate your manual command's flags
# Since cv::cvtColor is in 'imgproc', we target the essential dylibs directly
set(OPENCV_LIBS
    "/opt/homebrew/lib/libopencv_core.dylib"
    "/opt/homebrew/lib/libopencv_imgproc.dylib"
    "/opt/homebrew/lib/libopencv_highgui.dylib"
    "/opt/homebrew/lib/libopencv_imgcodecs.dylib"
    "/opt/homebrew/lib/libopencv_videoio.dylib"
    "/opt/homebrew/lib/libopencv_ximgproc.dylib"
)

set(SOURCE_FILES
    src/main.cpp
    src/ExtractGraph.cpp
)

add_executable(main ${SOURCE_FILES})

# 3. Include the headers (usually in this location for Brew)
target_include_directories(main PRIVATE "/opt/homebrew/include/opencv4")
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 4. Link against the absolute paths defined above
target_link_libraries(main PRIVATE ${OPENCV_LIBS})
