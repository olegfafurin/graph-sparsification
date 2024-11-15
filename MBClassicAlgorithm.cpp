//
// Created by Oleg Fafurin on 14.11.2024.
//

#include "MBClassicAlgorithm.h"
#include "MetricBackbone.h"


void MBClassicAlgorithm::calculate(const graph &g, MetricBackbone &mb) const {
    for (int i = 0; i < g.n; i++) {
        mb.add_SPT(i);
    }
}
