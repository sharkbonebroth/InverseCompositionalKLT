cmake_minimum_required(VERSION 3.15)
project(InverseCompositionalKLT)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(spdlog)

add_library(InverseCompositionalKLT
    ${CMAKE_CURRENT_SOURCE_DIR}/src/InverseCompositionalKLT.cpp
)

target_compile_features(InverseCompositionalKLT PUBLIC cxx_std_17)

target_link_libraries(InverseCompositionalKLT 
PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
PRIVATE
    spdlog::spdlog
)

target_include_directories(InverseCompositionalKLT
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(test)