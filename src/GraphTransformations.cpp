//
// Created by Oleg Fafurin on 04.12.2024.
//

#include "GraphTransformations.h"
#include <set>

graph contact_to_proximity(const graph& g_contact) {
    int N = g_contact.n;
    graph g_proximity(N);

    std::vector<std::vector<int>> neighbors(N);

    graph G_contact_copy = g_contact;
    for (int i = 0; i < N; ++i) {
        neighbors[i].push_back(i);
        for (const edge& e : g_contact.nodes[i].edges) {
            neighbors[i].push_back(e.v);
        }
        // Add self-loop with proximity=1 in G_proximity
        g_proximity.nodes[i].edges.emplace_back(i, i, 1.0);
        // Add self-loop with contact=1 in G_contact
        G_contact_copy.nodes[i].edges.emplace_back(i, i, 1.0);
    }
    g_proximity.m += N;

    for (int i = 0; i < N; ++i) {
        for (const edge& e : G_contact_copy.nodes[i].edges) {
            int j = e.v;
            if (i >= j) continue;

            std::set<int> cur_int;
            std::set<int> cur_union;

            for (int v : neighbors[i]) {
                cur_union.insert(v);
                if (std::find(neighbors[j].begin(), neighbors[j].end(), v) != neighbors[j].end()) {
                    cur_int.insert(v);
                }
            }
            for (int v : neighbors[j]) {
                cur_union.insert(v);
            }

            double num = 0.0, den = 0.0;
            for (int v : cur_int) {
                double weight_i_v = 0.0, weight_v_j = 0.0;
                for (const edge& e : G_contact_copy.nodes[i].edges) {
                    if (e.v == v) {
                        weight_i_v = e.w;
                        break;
                    }
                }
                for (const edge& e : G_contact_copy.nodes[v].edges) {
                    if (e.v == j) {
                        weight_v_j = e.w;
                        break;
                    }
                }
                num += std::min(weight_i_v, weight_v_j);
            }

            for (int v : cur_union) {
                double weight_i_v = 0.0, weight_v_j = 0.0;
                for (const edge& e : G_contact_copy.nodes[i].edges) {
                    if (e.v == v) {
                        weight_i_v = e.w;
                        break;
                    }
                }
                for (const edge& e : G_contact_copy.nodes[v].edges) {
                    if (e.v == j) {
                        weight_v_j = e.w;
                        break;
                    }
                }
                den += std::max(weight_i_v, weight_v_j);
            }

            if (den != 0.0) {
                long double proximity = num / den;
                g_proximity.nodes[i].edges.emplace_back(i, j, proximity);
                g_proximity.nodes[j].edges.emplace_back(j, i, proximity);
                g_proximity.m++;
            }
        }
    }

    return g_proximity;
}

graph proximity_to_distance(const graph &proximity_graph) {
    graph g(proximity_graph);
    g.path += "distance";

    for (auto &node : g.nodes) {
        for (auto &e: node.edges) {
            e.w = 1. / e.w - 1;
        }
    }
    for (int i = 0; i < g.nodes.size(); ++i) {
        auto &v = g.nodes[i].edges;
        v.erase(remove(v.begin(), v.end(), edge{i,i,0}), v.end());
        g.m--;
    }

    return g;
}
