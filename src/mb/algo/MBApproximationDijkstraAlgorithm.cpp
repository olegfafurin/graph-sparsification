//
// Created by Oleg Fafurin on 06.01.2025.
//

#include <random>
#include <vector>
#include "MBApproximationDijkstraAlgorithm.h"

#include <iostream>

using namespace std;

void MBApproximationDijkstraAlgorithm::calculate(const graph &g, MetricBackbone &mb) const {
    vector<int> nodesToTry;
    nodesToTry.reserve(g.n);
    for (int i = 0; i < g.n; i++) { nodesToTry.push_back(i); }

    int n_roots = n_roots_f(g.n);

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    shuffle(nodesToTry.begin(), nodesToTry.end(), rng);
    nodesToTry.erase(nodesToTry.begin() + n_roots, nodesToTry.end());

    if (verbose) {
        std::cout << "Dijkstra approximation started\nNumber of roots to build SPT from: " << n_roots << "\n";
    }

    for (int i = 0; i < nodesToTry.size(); i++) {
        auto s = nodesToTry[i];
        mb.add_SPT(s);
        // mb.writeToFile("test" + to_string(i) + ".txt");
    }
    if (verbose) {
        std::cout << "Dijkstra approximation finished\nNumber of edges found: " << mb.edges.size() << "\n";
    }
}