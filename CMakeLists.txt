cmake_minimum_required(VERSION 3.5)

project(streamKeying)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")

find_package(OpenCV 3.4.1 REQUIRED)
find_package(dlib 19.13 REQUIRED)


include_directories(include)
include_directories(${OpenCV_INCLUDE})
include_directories(${dlib_INCLUDE_DIRS})


# Test executable for desktop
set(CALIB_IMAGE_CREATOR keying)
add_executable(${CALIB_IMAGE_CREATOR} "src/main.cpp")
target_link_libraries(${CALIB_IMAGE_CREATOR} ${OpenCV_LIBS} ${dlib_LIBS} stdc++fs)
