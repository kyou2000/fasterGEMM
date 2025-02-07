cmake_minimum_required(VERSION 3.10)

project(gemmtest VERSION 1.0)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mfma")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")


include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(myLib src/sgemm.cpp src/tools.cpp)

add_executable(test1 src/testmatmul.cpp)

target_link_libraries(test1 myLib)

