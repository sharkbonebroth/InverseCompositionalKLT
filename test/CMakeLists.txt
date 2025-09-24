cmake_minimum_required(VERSION 3.15)

find_package(spdlog)

add_executable(testEuclideanKLT
    ${CMAKE_CURRENT_SOURCE_DIR}/testEuclideanKLT.cpp
)

target_compile_features(testEuclideanKLT PUBLIC cxx_std_17)

target_link_libraries(testEuclideanKLT
    spdlog::spdlog
    InverseCompositionalKLT
)