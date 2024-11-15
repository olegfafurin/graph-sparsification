cmake_minimum_required(VERSION 3.29)
project(code)

set(CMAKE_CXX_STANDARD 20)

add_executable(code
    MetricBackboneFast.cpp
        graph.h
        graph.cpp
        MetricBackbone.h
        MetricBackbone.cpp
        MBCalculationAlgorithm.h
        MBClassicAlgorithm.h
        MBClassicAlgorithm.cpp
        MBFast.h
        MBFast.cpp
)

#add_executable(gen-abcd gen_abcd.cpp graph.h graph.cpp
#        MetricBackbone.h
#        MetricBackbone.cpp
#        MBCalculationAlgorithm.h)
