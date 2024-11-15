//
// Created by Oleg Fafurin on 14.11.2024.
//

#ifndef MBFAST_H
#define MBFAST_H
#include "MBCalculationAlgorithm.h"


struct MBFast : MBCalculationAlgorithm {
    ~MBFast() override = default;

    void calculate(const graph &g, MetricBackbone &mb) const override;
};


#endif //MBFAST_H
