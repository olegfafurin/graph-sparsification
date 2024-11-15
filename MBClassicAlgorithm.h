//
// Created by Oleg Fafurin on 14.11.2024.
//

#ifndef MBCLASSICALGORITHM_H
#define MBCLASSICALGORITHM_H

#include "MBCalculationAlgorithm.h"
#include "MetricBackbone.h"

struct MBClassicAlgorithm : MBCalculationAlgorithm {
    ~MBClassicAlgorithm() override = default;

    void calculate(const graph &g, MetricBackbone &mb) const override;
};

#endif //MBCLASSICALGORITHM_H
