cmake_minimum_required (VERSION 2.8)

# project name
project (ssd_test)
# using C++11 
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")

# if u have OpenCV version more than one, set the build path which one u want to use
set(OpenCV_DIR "/home/mingcongchen/app/opencv-4.1.0/build")
# find opencv
find_package(OpenCV REQUIRED)
# print message of opencv
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

add_executable(ssd_test  main.cpp)

include_directories ( 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/include
    /usr/include 
${OpenCV_INCLUDE_DIRS})

target_link_libraries(ssd_test
${OpenCV_LIBS}
/usr/lib
/usr/local/lib
    )
