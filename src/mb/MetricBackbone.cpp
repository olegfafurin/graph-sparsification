//
// Created by Oleg Fafurin on 14.11.2024.
//

#include <fstream>
#include "MetricBackbone.h"

#include "mb/algo/MBClassicAlgorithm.h"

using namespace std;


MetricBackbone::MetricBackbone(const graph &g, MBCalculationAlgorithm &algorithm): base_graph(g), algorithm(algorithm) {
    compute();
}

MetricBackbone::MetricBackbone(const MetricBackbone &other): base_graph(other.base_graph), algorithm(other.algorithm),
                                                             edges(other.edges) {
}

void MetricBackbone::add_SPT(const int &root) {
    std::vector<std::pair<int, double> > parent(base_graph.n, {-1, -1});
    base_graph.getSPT(root, parent, base_graph.nodes);
    for (int u = 0; u < base_graph.n; u++) {
        auto [v, w] = parent[u];
        if (v == -1) { continue; }
        edges[{min(u, v), max(u, v)}] = w;
    }
}

void MetricBackbone::writeToFile(const std::string &filename, const std::function<double(double)> &transform,
                                 bool write_header) const {
    ofstream ofs(filename);
    ofs << base_graph.n << " " << edges.size() << '\n';
    for (auto &[p, w]: edges) {
        ofs << p.first << " " << p.second << " " << transform(w) << '\n';
    }
    ofs.close();
}

void MetricBackbone::compute() {
    MetricBackbone &backbone = *this;
    algorithm.calculate(base_graph, backbone);
}
