cmake_minimum_required(VERSION 3.19)
project(Pathfinding)

set(CMAKE_CXX_STANDARD 20)

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
    "/opt/homebrew/lib/libopencv_video.dylib"
    "/opt/homebrew/lib/libopencv_ximgproc.dylib"
    "/opt/homebrew/lib/libopencv_geometry.dylib"
)

add_library(hungarian-algorithm-cpp STATIC
  extern/hungarian-algorithm-cpp/Hungarian.cpp
  extern/hungarian-algorithm-cpp/Hungarian.h
)

set(SOURCE_FILES
    src/main.cpp
    src/ExtractGraph.cpp
)

set(TEST_FILES
    src/ExtractGraph.cpp
    src/test_video.cpp
)

add_executable(main ${SOURCE_FILES})

enable_testing()
add_executable(path_tests ${TEST_FILES})
add_test(NAME TestRaw1 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/raw-1.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/raw-1.mp4")
add_test(NAME TestRaw2 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/raw-2.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/raw-2.mp4")
add_test(NAME TestRaw3 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/raw-3.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/raw-3.mp4")
add_test(NAME TestRaw4 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/raw-4.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/raw-4.mp4")

add_test(NAME TestLine1 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-1.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-1.mp4")
add_test(NAME TestLine2 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-2.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-2.mp4")
add_test(NAME TestLine3 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-3.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-3.mp4")
add_test(NAME TestLine4 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-4.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-4.mp4")
add_test(NAME TestLine5 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-5.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-5.mp4")
add_test(NAME TestLine6 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-6.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-6.mp4")
add_test(NAME TestLine7 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-7.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-7.mp4")
add_test(NAME TestLine8 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-8.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-8.mp4")
add_test(NAME TestLine9 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/line-9.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/line-9.mp4")

add_test(NAME TestGreen1 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/green-1.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/green-1.mp4")
add_test(NAME TestGreen2 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/green-2.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/green-2.mp4")
add_test(NAME TestGreen3 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/green-3.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/green-3.mp4")
add_test(NAME TestGreen4 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/green-4.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/green-4.mp4")
add_test(NAME TestGreen5 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/green-5.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/green-5.mp4")
add_test(NAME TestGreen6 COMMAND path_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/green-6.mp4" "${CMAKE_CURRENT_SOURCE_DIR}/output/green-6.mp4")

# 3. Include the headers (usually in this location for Brew)
target_include_directories(main PRIVATE "/opt/homebrew/include/opencv5")
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/hungarian-algorithm-cpp)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 4. Link against the absolute paths defined above
target_link_libraries(main PRIVATE ${OPENCV_LIBS} ${OpenCV_LIBRARIES} hungarian-algorithm-cpp)

# 3. Include the headers (usually in this location for Brew)
target_include_directories(path_tests PRIVATE "/opt/homebrew/include/opencv5")
target_include_directories(path_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(path_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/hungarian-algorithm-cpp)
target_include_directories(path_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 4. Link against the absolute paths defined above
target_link_libraries(path_tests PRIVATE ${OPENCV_LIBS} ${OpenCV_LIBRARIES} hungarian-algorithm-cpp)
