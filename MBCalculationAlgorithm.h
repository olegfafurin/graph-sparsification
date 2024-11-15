//
// Created by Oleg Fafurin on 14.11.2024.
//

#ifndef MBCALCULATIONALGORITHM_H
#define MBCALCULATIONALGORITHM_H

#include "graph.h"
#include "MetricBackbone.h"


struct MBCalculationAlgorithm {
    MBCalculationAlgorithm() {};

    virtual ~MBCalculationAlgorithm() = default;

    virtual void calculate(const graph &g, MetricBackbone &mb) const {}
};

#endif //MBCALCULATIONALGORITHM_H
