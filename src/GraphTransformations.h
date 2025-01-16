//
// Created by Oleg Fafurin on 04.12.2024.
//

#ifndef GRAPHTRANSFORMATIONS_H
#define GRAPHTRANSFORMATIONS_H

#include "graph.h"

graph contact_to_proximity(const graph &contact_graph);

graph proximity_to_distance(const graph &proximity_graph);

#endif //GRAPHTRANSFORMATIONS_H
