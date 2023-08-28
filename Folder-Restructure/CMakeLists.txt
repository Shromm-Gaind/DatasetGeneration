cmake_minimum_required(VERSION 3.25)
project(PigMMdetection3D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the required package
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(PigMMdetection3D main.cpp)

# Link the libraries
target_link_libraries(PigMMdetection3D ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

