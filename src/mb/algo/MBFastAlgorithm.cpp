//
// Created by Oleg Fafurin on 14.11.2024.
//


#include <vector>
#include <queue>
#include <set>
#include <iostream>
#include "MBFastAlgorithm.h"
#include "MBFastCommon.h"

using namespace std;

set<edge> getMBFirstStage(graph &g);

set<edge> getMBFast(const graph &g, bool verbose);

bool getMetricBFS(const graph &g, const edge &e);

void MBFastAlgorithm::calculate(const graph &g, MetricBackbone &mb) const {
    auto e = getMBFast(g, verbose);
    for (auto &[u,v,w]: e) {
        mb.edges[{min(u,v), max(u,v)}] = w;
    }
}



set<edge> getMBFast(const graph &g, bool verbose = false) {
    set<edge> edges = getMBFirstStage(g);
    if (verbose) {
        cout << "base graph edge count: " << g.m << endl;
        cout << "possibly metric edges count after step 1: " << edges.size() << '\n';
    }

    vector<priority_queue<edge, vector<edge>, greater<> > > vi(g.n);
    for (auto &e: edges) {
        vi[e.u].emplace(e.u, e.v, e.w);
        vi[e.v].emplace(e.v, e.u, e.w);
    }
    set<edge> ms;
    for (int i = 0; i < g.n; i++) {
        set<edge> ms_local;
        priority_queue local_queue(vi[i]);
        if (!vi[i].empty()) {
            auto least_w = local_queue.top().w;
            while (!local_queue.empty()) {
                if (least_w == local_queue.top().w) {
                    ms_local.insert(local_queue.top());
                    local_queue.pop();
                    continue;
                }
                edge e = local_queue.top();
                local_queue.pop();
                for (auto &edge: ms_local) {
                    if (e.w >= edge.w + vi[edge.v].top().w) {
                        goto vertex_considered;
                    }
                }
                ms_local.insert(e);
            }
        }
    vertex_considered:
        ms.merge(ms_local);
    }

    if (verbose) {
        cout << "metric edges after step 2: size " << ms.size() << '\n';
    }

    set<edge> u;
    set_difference(edges.begin(), edges.end(), ms.begin(), ms.end(), std::inserter(u, u.end()));

    for (auto &e: u) {
        if (getMetricBFS(g, e)) {
            ms.insert(e);
        }
    }

    if (verbose) {
        cout << "metric edges found: " << ms.size() << '\n';
    }

    return ms;
}

bool getMetricBFS(const graph &g, const edge &e) {
    int u = e.u, v = e.v;
    double d = e.w;
    vector dist(g.n, 1e15);
    vector visited(g.n, false);
    priority_queue<pair<double, int>, vector<pair<double, int> >, greater<> > pq;
    dist[u] = 0ll;
    pq.emplace(0, u);

    while (!pq.empty()) {
        int cur = pq.top().second;
        if (cur == v && pq.top().first < d) {
            return false;
        }
        pq.pop();
        if (visited[cur]) { continue; }
        visited[cur] = true;

        for (auto &x: g.nodes[cur].edges)
            if (dist[x.v] > dist[cur] + x.w) {
                dist[x.v] = dist[cur] + x.w;
                pq.emplace(dist[x.v], x.v);
            }
    }
    return true;
}
