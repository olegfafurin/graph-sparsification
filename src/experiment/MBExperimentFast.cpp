#include <random>
#include <iostream>
#include <fstream>

#include "graph.h"
#include "mb/algo/MBClassicAlgorithm.h"
#include "mb/algo/MBFastAlgorithm.h"
#include "mb/MetricBackbone.h"


using namespace std;
typedef long long ll;


void compare_time(const string &path) {
    graph G(path);
    cout << "#edges: " << G.m << '\n';
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // G.getMB(false);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    cout << "old computed\n";

    auto t_old = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    // set<edge> mb = G.getMBFast();
    end = std::chrono::steady_clock::now();

    cout << "new computed\n";

    auto t_new = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    // G = graph(path);
    begin = std::chrono::steady_clock::now();
    // G.getMBFirstStageAndBFS();
    end = std::chrono::steady_clock::now();

    auto t_first_stage_plus_bfs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    cout << "Old time: " << t_old << " μs" << '\n';
    cout << "New time: " << t_new << " μs" << '\n';
    cout << "Time first stage + bfs: " << t_first_stage_plus_bfs << " μs" << '\n';
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " graph_filename" << '\n';
    }
    string filename = argv[1];

    cout << "reading graph: " << filename << '\n';

    graph G(filename);
    auto fast_algo = MBFastAlgorithm(true);
    // auto noop_algo = MBCalculationAlgorithm();
    MBCalculationAlgorithm &algorithm{fast_algo};
    // MBCalculationAlgorithm &alg_classic{classic_algo};
    // MBCalculationAlgorithm base_alg = algorithm;

    MetricBackbone mb(G, algorithm);

    // vector<int> nodesToTry;
    // nodesToTry.reserve(G.n);
    // for (int i = 0; i < G.n; i++) { nodesToTry.push_back(i); }
    //
    // auto rd = std::random_device{};
    // auto rng = std::default_random_engine{rd()};
    // shuffle(nodesToTry.begin(), nodesToTry.end(), rng);
    // nodesToTry.erase(nodesToTry.begin() + 4, nodesToTry.end());
    //
    // for (int i = 0; i < nodesToTry.size(); i++) {
    //     auto s = nodesToTry[i];
    //     mb.add_SPT(s);
    //     mb.writeToFile("test" + to_string(i) + ".txt");
    // }

    mb.writeToFile("test_fast_mb.txt");

    // MetricBackbone mb_fast = MetricBackbone(G, algorithm);
    // mb_fast.writeToFile("test_fast_mb.txt");


    // G.writeMBToFile(false);

    // graph G("Datasets/sbm/sbm_n=10_p1=05,p2=01.txt");


    // ofstream ofs("output/MB_Datasets/USairport500-uniform-fast.txt");
    // ofs << G.n << " " << metric.size() << std::endl;
    // for (auto &e: metric) {
    //     ofs << e.u << " " << e.v << " " << e.w << '\n';
    // }
    // ofs.close();
}
