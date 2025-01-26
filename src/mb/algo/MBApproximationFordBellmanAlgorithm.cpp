//
// Created by Oleg Fafurin on 19.01.2025.
//

#include <iostream>
#include "MBApproximationFordBellmanAlgorithm.h"


void MBApproximationFordBellmanAlgorithm::calculate(const graph &g, MetricBackbone &mb) const {
    int max_n_step = max_steps_f(g.n);
    if (verbose) {
        std::cout << "Ford-Bellman approximation started\nSemi-metricity order explored: " << max_n_step << "\n";
    }
    for (int i = 0; i < g.n; ++i) {
        std::vector<std::pair<int, double> > parent(g.n, {-1, -1});
        g.getSPTFordBellman(i, parent, max_n_step);
        for (auto e: g.nodes[i].edges) {
            if (parent[e.v].first == i) {
                mb.edges[{std::min(e.v, i), std::max(e.v, i)}] = e.w;
            }
        }
        if (verbose && i % 100 == 0) {
            std::cout << "Ford-Bellman approximation: added vertex " << i << "\n";
        }
    }
    if (verbose) {
        std::cout << "Ford-Bellman approximation finished\nNumber of edges found: " << mb.edges.size() << "\n";
    }
}
