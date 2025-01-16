//
// Created by Oleg Fafurin on 06.01.2025.
//

#include <random>
#include <vector>
#include "MBApproximationAlgorithm.h"

using namespace std;

void MBApproximationAlgorithm::calculate(const graph &g, MetricBackbone &mb) const {
    vector<int> nodesToTry;
    nodesToTry.reserve(g.n);
    for (int i = 0; i < g.n; i++) { nodesToTry.push_back(i); }

    int n_roots = 4 * log(g.n);

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    shuffle(nodesToTry.begin(), nodesToTry.end(), rng);
    nodesToTry.erase(nodesToTry.begin() + n_roots, nodesToTry.end());

    for (int i = 0; i < nodesToTry.size(); i++) {
        auto s = nodesToTry[i];
        mb.add_SPT(s);
        mb.writeToFile("test" + to_string(i) + ".txt");
    }
}