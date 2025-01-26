//
// Created by Oleg Fafurin on 02.12.2024.
//

#include <set>
#include <iostream>
#include "MBFirstAndBFSAlgorithm.h"
#include "MBFastCommon.h"
#include "graph.h"

using namespace std;

void MBFirstAndBFSAlgorithm::calculate(const graph &g, MetricBackbone &mb) const {
    set<edge> edges = getMBFirstStage(g);
    if (verbose) {
        cout << "base graph edge count: " << g.m << endl;
        cout << "possibly metric edges count after step 1: " << edges.size() << '\n';
    }

    vector<node> candidate_adj_list(g.n);
    for (auto &e: edges) {
        candidate_adj_list[e.u].edges.emplace_back(e.u, e.v, e.w);
        candidate_adj_list[e.v].edges.emplace_back(e.v, e.u, e.w);
    }
    for (int i = 0; i < g.n; ++i) {
        vector<pair<int, double> > parent(g.n, {-1, -1});
        g.getSPTDijkstra(i, parent, candidate_adj_list);
        for (int u = 0; u < g.n; u++) {
            auto [v, w] = parent[u];
            if (v == -1) { continue; }
            mb.edges[{min(u, v), max(u, v)}] = w;
        }
    }
    if (verbose) {
        cout << "metric edges count: " << mb.edges.size() << '\n';
    }
}
