cmake_minimum_required(VERSION 3.29)
project(code)

set(CMAKE_CXX_STANDARD 20)

add_executable(first-and-bfs
        src/experiment/MBExperimentFirstAndBFS.cpp
        src/graph.h
        src/graph.cpp
        src/mb/MetricBackbone.h
        src/mb/MetricBackbone.cpp
        src/mb/algo/MBCalculationAlgorithm.h
        src/mb/algo/MBFirstAndBFSAlgorithm.h
        src/mb/algo/MBFirstAndBFSAlgorithm.cpp
        src/mb/algo/MBFastCommon.h
)

add_executable(fast
        src/experiment/MBExperimentFast.cpp
        src/graph.h
        src/graph.cpp
        src/mb/MetricBackbone.h
        src/mb/MetricBackbone.cpp
        src/mb/algo/MBCalculationAlgorithm.h
        src/mb/algo/MBFastAlgorithm.h
        src/mb/algo/MBFastAlgorithm.cpp
        src/mb/algo/MBFastCommon.h
)

add_executable(classic
        src/experiment/MBExperimentClassic.cpp
        src/graph.h
        src/graph.cpp
        src/mb/MetricBackbone.h
        src/mb/MetricBackbone.cpp
        src/mb/algo/MBCalculationAlgorithm.h
        src/mb/algo/MBClassicAlgorithm.h
        src/mb/algo/MBClassicAlgorithm.cpp
        src/mb/algo/MBFastAlgorithm.h
        src/mb/algo/MBFastAlgorithm.cpp
        src/GraphTransformations.cpp
        src/GraphTransformations.h
)

add_executable(compare-time
        src/experiment/MBExperimentCompareTime.cpp
        src/graph.h
        src/graph.cpp
        src/mb/MetricBackbone.h
        src/mb/MetricBackbone.cpp
        src/mb/algo/MBCalculationAlgorithm.h
        src/mb/algo/MBClassicAlgorithm.h
        src/mb/algo/MBClassicAlgorithm.cpp
        src/mb/algo/MBFastAlgorithm.h
        src/mb/algo/MBFastAlgorithm.cpp
        src/mb/algo/MBFirstAndBFSAlgorithm.h
        src/mb/algo/MBFirstAndBFSAlgorithm.cpp
        src/mb/algo/MBApproximationDijkstraAlgorithm.h
        src/mb/algo/MBApproximationDijkstraAlgorithm.cpp
        src/mb/algo/MBApproximationFordBellmanAlgorithm.cpp
        src/mb/algo/MBApproximationFordBellmanAlgorithm.h
)

add_executable(ford-bellman
        src/experiment/MBExperimentFordBellmanApprox.cpp
        src/graph.h
        src/graph.cpp
        src/mb/MetricBackbone.h
        src/mb/MetricBackbone.cpp
        src/mb/algo/MBCalculationAlgorithm.h
        src/mb/algo/MBApproximationFordBellmanAlgorithm.cpp
        src/mb/algo/MBApproximationFordBellmanAlgorithm.h
)

add_executable(num-test
        tmp/num_experiments.cpp
)

add_executable(graph-transform-test
        tmp/graph-transform-test.cpp
        src/graph.h
        src/graph.cpp
        src/GraphTransformations.h
        src/GraphTransformations.cpp
)

add_executable(prepare-approx
        src/experiment/MBExperimentPrepareApproximations.cpp
        src/graph.h
        src/graph.cpp
        src/GraphTransformations.h
        src/GraphTransformations.cpp
        src/mb/MetricBackbone.h
        src/mb/MetricBackbone.cpp
        src/mb/algo/MBCalculationAlgorithm.h
        src/mb/algo/MBClassicAlgorithm.h
        src/mb/algo/MBClassicAlgorithm.cpp
)

target_include_directories(classic PRIVATE src)
target_include_directories(fast PRIVATE src)
target_include_directories(first-and-bfs PRIVATE src)
target_include_directories(compare-time PRIVATE src)
target_include_directories(graph-transform-test PRIVATE src)
target_include_directories(prepare-approx PRIVATE src)
target_include_directories(ford-bellman PRIVATE src)



#add_executable(gen-abcd gen_abcd.cpp graph.h graph.cpp
#        MetricBackbone.h
#        MetricBackbone.cpp
#        MBCalculationAlgorithm.h)
