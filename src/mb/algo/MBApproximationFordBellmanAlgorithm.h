//
// Created by Oleg Fafurin on 19.01.2025.
//

#ifndef MBAPPROXIMATIONFORDBELLMANALGORITHM_H
#define MBAPPROXIMATIONFORDBELLMANALGORITHM_H

#include "MBCalculationAlgorithm.h"


struct MBApproximationFordBellmanAlgorithm : MBCalculationAlgorithm {
    const std::function<int(int)> max_steps_f;
    bool verbose;

    explicit MBApproximationFordBellmanAlgorithm(const std::function<int(int)> &max_steps_f,
                                                 bool verbose = false): max_steps_f(max_steps_f), verbose(verbose) {
    }

    ~MBApproximationFordBellmanAlgorithm() override = default;

    void calculate(const graph &g, MetricBackbone &mb) const override;
};


#endif //MBAPPROXIMATIONFORDBELLMANALGORITHM_H
