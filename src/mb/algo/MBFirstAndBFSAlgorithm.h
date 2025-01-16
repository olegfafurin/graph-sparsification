//
// Created by Oleg Fafurin on 02.12.2024.
//

#ifndef MBFIRSTANDBFSALGORITHM_H
#define MBFIRSTANDBFSALGORITHM_H

#include "MBCalculationAlgorithm.h"


struct MBFirstAndBFSAlgorithm: MBCalculationAlgorithm {
    bool verbose;

    MBFirstAndBFSAlgorithm(bool verbose = false): verbose(verbose) {
    }

    ~MBFirstAndBFSAlgorithm() override = default;

    void calculate(const graph &g, MetricBackbone &mb) const override;
};


#endif //MBFIRSTANDBFSALGORITHM_H
