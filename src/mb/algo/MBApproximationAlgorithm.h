//
// Created by Oleg Fafurin on 06.01.2025.
//

#ifndef MBAPPROXIMATIONALGORITHM_H
#define MBAPPROXIMATIONALGORITHM_H
#include "MBCalculationAlgorithm.h"


class MBApproximationAlgorithm : MBCalculationAlgorithm {
public:
    ~MBApproximationAlgorithm() override = default;

    void calculate(const graph &g, MetricBackbone &mb) const override;
};


#endif //MBAPPROXIMATIONALGORITHM_H
