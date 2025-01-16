//
// Created by Oleg Fafurin on 02.12.2024.
//

#ifndef MBFASTCOMMON_H
#define MBFASTCOMMON_H

#include <set>
#include "graph.h"

using namespace std;

inline set<edge> getMBFirstStage(const graph &g) {
    set<edge> edges;
    for (int i = 0; i < g.n; i++) {
        for (auto &e: g.nodes[i].edges) {
            if (e.u < e.v) {
                edges.insert({e.u, e.v, e.w});
            }
        }
    }
    set<edge> res;
    for (auto &e: edges) {
        bool take = true;
        int u = e.u, v = e.v;
        for (auto &x: g.nodes[u].edges) {
            for (auto &y: g.nodes[v].edges) {
                if (x.v == y.v && x.w + y.w < e.w) {
                    take = false;
                    goto loop_end;
                }
            }
        }
    loop_end:
        if (take) {
            res.insert({u, v, e.w});
        }
    }
    return res;
}


#endif //MBFASTCOMMON_H
