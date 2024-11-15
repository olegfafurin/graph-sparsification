#include <random>
#include <iostream>
#include <fstream>

#include "graph.h"
#include "MBClassicAlgorithm.h"
#include "MetricBackbone.h"


using namespace std;
typedef long long ll;


// bool compare(const string &path) {
//     graph G(path);
//     set<edge> mb_fast = G.getMBFast();
//     G.getMB(false);
//
//     bool flag = true;
//     for (auto &e: mb_fast) {
//         if (!G.edges.contains({e.u, e.v})) {
//             flag = false;
//             cout << "edge " << e.u << " " << e.v << " not found in original MB" << '\n';
//         }
//     }
//     return flag;
// }


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

int main() {

    graph G("Datasets/abcd/n=1000/edge.txt");
    auto classic_alg = MBClassicAlgorithm();
    MBCalculationAlgorithm &algorithm{classic_alg};

    MBCalculationAlgorithm base_alg = MBCalculationAlgorithm();

    MetricBackbone mb(G, base_alg);

    vector<int> nodesToTry;
    nodesToTry.reserve(G.n);
    for (int i = 0; i < G.n; i++) { nodesToTry.push_back(i); }

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    shuffle(nodesToTry.begin(), nodesToTry.end(), rng);
    nodesToTry.erase(nodesToTry.begin() + 5, nodesToTry.end());

    for (auto s: nodesToTry) {
        mb.add_SPT(s);
        mb.write_to_file("test" + to_string(s) + ".txt");
    }

    // G.writeMBToFile(false);

    // graph G("Datasets/sbm/sbm_n=10_p1=05,p2=01.txt");


    // ofstream ofs("output/MB_Datasets/USairport500-uniform-fast.txt");
    // ofs << G.n << " " << metric.size() << std::endl;
    // for (auto &e: metric) {
    //     ofs << e.u << " " << e.v << " " << e.w << '\n';
    // }
    // ofs.close();
}
