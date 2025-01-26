//
// Created by Oleg Fafurin on 06.01.2025.
//

#ifndef MBAPPROXIMATIONALGORITHM_H
#define MBAPPROXIMATIONALGORITHM_H
#include "MBCalculationAlgorithm.h"


struct MBApproximationDijkstraAlgorithm : MBCalculationAlgorithm {
    const std::function<int(int)> n_roots_f;

    bool verbose;

    explicit MBApproximationDijkstraAlgorithm(const std::function<int(int)> &n_roots,
                                              bool verbose = false) : n_roots_f(n_roots), verbose(verbose) {
    }

    ~MBApproximationDijkstraAlgorithm() override = default;

    void calculate(const graph &g, MetricBackbone &mb) const override;
};


#endif //MBAPPROXIMATIONALGORITHM_H
