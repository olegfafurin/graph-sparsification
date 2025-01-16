//
// Created by Oleg Fafurin on 14.11.2024.
//

#include "MBClassicAlgorithm.h"

#include <iostream>


MBClassicAlgorithm::MBClassicAlgorithm(bool verbose): verbose(verbose) {
}

void MBClassicAlgorithm::calculate(const graph &g, MetricBackbone &mb) const {
    if (verbose) {
        std::cout << "constructing MB for a graph with " << g.n << " verices, " << g.m << " edges\n";
    }
    for (int i = 0; i < g.n; i++) {
        mb.add_SPT(i);
        if (i % 100 == 0 && verbose) {
            std::cout << "added SPT "  << i << "; MB edge count: " << mb.edges.size() << '\n';
        }
    }
}
