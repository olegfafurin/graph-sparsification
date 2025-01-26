#include <random>
#include <iostream>
#include <fstream>

#include "graph.h"
#include "mb/algo/MBClassicAlgorithm.h"
#include "mb/algo/MBFastAlgorithm.h"
#include "mb/MetricBackbone.h"
#include "mb/algo/MBApproximationDijkstraAlgorithm.h"
#include "mb/algo/MBApproximationFordBellmanAlgorithm.h"
#include "mb/algo/MBFirstAndBFSAlgorithm.h"


using namespace std;
typedef long long ll;


void compare_time(const string &graph_filename) {
    graph G(graph_filename);

    auto algo1 = MBFastAlgorithm(true);
    auto algo2 = MBClassicAlgorithm(true);
    auto algo3 = MBFirstAndBFSAlgorithm(true);
    auto algo4 = MBApproximationDijkstraAlgorithm([](int n) { return 2*log2(n); }, true);
    auto algo5 = MBApproximationFordBellmanAlgorithm([](int n) { return std::max(2, 2 * static_cast<int>(log2(n))); }, true);


    MBCalculationAlgorithm &fast_algo{algo1};
    MBCalculationAlgorithm &classic_algo{algo2};
    MBCalculationAlgorithm &first_and_bfs_algo{algo3};
    MBCalculationAlgorithm &dijkstra_approx_algo{algo4};
    MBCalculationAlgorithm &fb_approx_algo{algo5};


    cout << "#vertices: " << G.n << '\n';
    cout << "#edges: " << G.m << '\n';

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    MetricBackbone mb_clasic(G, classic_algo);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    cout << "classic computed\n";

    auto t_classic = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    MetricBackbone mb(G, fast_algo);
    end = std::chrono::steady_clock::now();

    cout << "fast computed\n";

    auto t_fast = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    MetricBackbone mb_first_and_bfs_algo(G, first_and_bfs_algo);
    end = std::chrono::steady_clock::now();

    auto t_first_and_bfs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    MetricBackbone mb_dijkstra_approx(G, dijkstra_approx_algo);
    end = std::chrono::steady_clock::now();
    auto t_dijkstra_approx = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();


    begin = std::chrono::steady_clock::now();
    MetricBackbone mb_ford_bellman_algo(G, fb_approx_algo);
    end = std::chrono::steady_clock::now();
    auto t_ford_bellman = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    string filename_no_ext = graph_filename.substr(0, graph_filename.find_last_of('.'));

    ofstream ofs(filename_no_ext + "_time_mb.csv");
    ofs << "t_apsp,t_ksl,t_1_and_apsp,t_dijkstra_log2,t_fb_log2\n";
    ofs << t_classic << ',' << t_fast << ',' << t_first_and_bfs << ',' << t_dijkstra_approx  << ',' << t_ford_bellman << '\n';
    ofs.close();

    cout << "Classic (union of APSPs) algo time: " << t_classic << " μs" << '\n';
    cout << "Fast (3-stage) algo time: " << t_fast << " μs" << '\n';
    cout << "First stage + classic: " << t_first_and_bfs << " μs" << '\n';
    cout << "Dijkstra approximation with log2(n) SPTs: " << t_dijkstra_approx << " μs" << '\n';
    cout << "Ford-Bellman approximation with log2(n) depth: " << t_ford_bellman << " μs" << '\n';
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <graph_filename>" << '\n';
    }
    string graph_filename = argv[1];

    compare_time(graph_filename);
}
