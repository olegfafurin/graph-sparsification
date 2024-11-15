//
// Created by Oleg Fafurin on 14.11.2024.
//

#include <fstream>
#include "MetricBackbone.h"

#include "MBClassicAlgorithm.h"

using namespace std;


// MetricBackbone::MetricBackbone(const graph &g): base_graph(g), algorithm(MBCalculationAlgorithm()) {
//     compute();
// }

MetricBackbone::MetricBackbone(const graph &g, MBCalculationAlgorithm &algorithm): base_graph(g), algorithm(algorithm) {
    compute();
}

void MetricBackbone::add_SPT(const int &root) {
    std::vector<std::pair<int, double>> parent(base_graph.n, {-1, -1});
    base_graph.getSPT(root, parent, base_graph.nodes);
    for (int u = 0; u < base_graph.n; u++) {
        auto [v, w] = parent[u];
        if (v == -1) { continue; }
        edges[{min(u, v), max(u, v)}] = w;
    }
}

void MetricBackbone::write_to_file(const std::string &filename) const {
    ofstream ofs(filename);
    // ofs << n << " " << edgesMB.size() << '\n';
    for (auto &[p, w]: edges) {
        ofs << p.first << " " << p.second << '\n';
    }
    ofs.close();
}

void MetricBackbone::compute() {
    MetricBackbone &backbone = *this;
    algorithm.calculate(base_graph, backbone);
    // for (auto &p: mb_set) {
    //     edges[{min(p.u, p.v), max(p.u, p.v)}] = p.w;
    // }
}
