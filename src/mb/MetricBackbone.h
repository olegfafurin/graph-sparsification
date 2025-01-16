//
// Created by Oleg Fafurin on 14.11.2024.
//

#ifndef METRICBACKBONECALCULATOR_H
#define METRICBACKBONECALCULATOR_H

#include "graph.h"

struct MBCalculationAlgorithm;

struct MetricBackbone {
    std::unordered_map<std::pair<int, int>, double, pair_hash> edges;

    const graph &base_graph;

    const MBCalculationAlgorithm &algorithm;

    // explicit MetricBackbone(const graph &g);

    explicit MetricBackbone(const graph &g, MBCalculationAlgorithm &algorithm);

    MetricBackbone(const MetricBackbone &other);

    void add_SPT(const int &root);

    void writeToFile(const std::string &filename,
                     const std::function<double(double)> &transform = [](double _w) { return _w; },
                     bool write_header = true) const;

private:
    void compute();
};


#endif //METRICBACKBONECALCULATOR_H
