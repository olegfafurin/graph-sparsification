//
// Created by Oleg Fafurin on 06.01.2025.
//


#include <random>
#include <iostream>
#include "graph.h"
#include "mb/algo/MBClassicAlgorithm.h"
#include "mb/algo/MBFastAlgorithm.h"
#include "mb/algo/MBFirstAndBFSAlgorithm.h"
#include "mb/MetricBackbone.h"
#include "GraphTransformations.h"


using namespace std;
typedef long long ll;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <graph_filename>" << '\n';
    }
    string graph_filename = argv[1];
    string filename_no_ext = graph_filename.substr(0, graph_filename.find_last_of('.'));
    cout << "reading graph: " << graph_filename << '\n';

    auto classic_algo = MBClassicAlgorithm(true);
    MBCalculationAlgorithm &alg_classic{classic_algo};

    graph G(graph_filename); // unweighted, so it is its own MB
    int n = G.n;

    cout << "graph is read, constructing the proximity graph\n";

    graph G_proximity = contact_to_proximity(G);

    G_proximity.writeToFile(filename_no_ext + "_w_only.txt", false);
    G_proximity.writeToFile(filename_no_ext + "_w.txt", true);
    cout << "weighted proximity graph is constructed, constructing distance graph\n";

    graph G_distance = proximity_to_distance(G_proximity);

    cout << "distance graph is constructed, constructing MB\n";

    MetricBackbone mb_distance(G_distance, alg_classic);

    cout << "MB for distance graph is constructed\n";

    mb_distance.writeToFile(filename_no_ext + "_proximity_mb_full.txt", [](auto w) { return 1 / (w + 1); });

    vector<int> nodesToTry;
    nodesToTry.reserve(n);
    for (int i = 0; i < n; i++) { nodesToTry.push_back(i); }

    int n_roots = n;

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    shuffle(nodesToTry.begin(), nodesToTry.end(), rng);
    nodesToTry.erase(nodesToTry.begin() + n_roots, nodesToTry.end());


    MBCalculationAlgorithm none_alg;
    MetricBackbone mb_distance_approx(G_distance, none_alg);

    for (int i = 0; i < nodesToTry.size(); i++) {
        auto s = nodesToTry[i];
        mb_distance_approx.add_SPT(s);
        mb_distance_approx.writeToFile(filename_no_ext + "_proximity_mb_approx_" + to_string(i + 1) + ".txt",
                                       [](auto w) { return 1 / (w + 1); });
    }
}
