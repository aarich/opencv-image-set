cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
 
project(connect_kinect)

find_package( catkin REQUIRED COMPONENTS roscpp )
find_package( OpenCV REQUIRED )
find_package(Boost REQUIRED)

add_executable(matching_to_many_images matching_to_many_images.cpp)

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

target_link_libraries (matching_to_many_images ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} /opt/ros/hydro/lib/libcv_bridge.so)
