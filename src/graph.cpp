//
// Created by Oleg Fafurin on 13.11.2024.
//


#include "graph.h"
#include <unordered_set>
#include <set>
#include <random>
#include <fstream>
#include <queue>

using namespace std;


edge::edge(int _u, int _v, double _w): u(_u), v(_v), w(_w) {
}

graph::graph(const string &_path) : path(_path) {
    ifstream ifs(path);
    ifs >> n >> m;
    nodes.resize(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        double w;
        ifs >> u >> v >> w;
        add_edge(u, v, w);
    }
    ifs.close();
}

graph::graph(const graph &g): n(g.n), m(g.m), path(g.path), nodes(g.nodes) {
}

graph::graph(int n): n(n), m(0), nodes(n) {
}

void graph::add_edge(int u, int v, double w) {
    nodes[u].edges.emplace_back(u, v, w);
    nodes[v].edges.emplace_back(v, u, w);
}

void graph::getSPT(int s, std::vector<std::pair<int, double> > &parent,
                   const std::vector<node> &vertexEdges) const {
    vector<double> dist(n, 1e15);
    vector<bool> visited(n, false);
    priority_queue<pair<double, int>, vector<pair<double, int> >, greater<> > pq;
    dist[s] = 0ll;
    pq.emplace(0, s);

    while (!pq.empty()) {
        int v = pq.top().second;
        double d = pq.top().first;

        pq.pop();
        if (dist[v] < d)
            continue;

        for (auto &e: vertexEdges[v].edges)
            if (dist[e.v] > dist[v] + e.w) {
                dist[e.v] = dist[v] + e.w;
                parent[e.v] = {e.u, e.w};
                pq.emplace(dist[e.v], e.v);
            }
    }
}

void graph::writeToFile(const std::string &filename, bool include_header) {
    ofstream ofs(filename);
    if (include_header)
        ofs << n << " " << m << '\n';
    for (int i = 0; i < n; i++) {
        for (auto &edge: nodes[i].edges) {
            if (edge.v < i)
                continue;
            ofs << edge.u << " " << edge.v << " " << edge.w << "\n";
        }
    }
    ofs.close();
}

bool operator<(edge const &x, edge const &y) {
    if (x.w < y.w)
        return true;
    if (x.w == y.w) {
        if (min(x.v, x.u) < min(y.v, y.u))
            return true;
        if (min(x.v, x.u) == min(y.v, y.u) && max(x.v, x.u) < max(y.v, y.u))
            return true;
    }
    return false;
}

bool operator>(edge const &x, edge const &y) {
    if (x.w > y.w)
        return true;
    if (x.w == y.w) {
        if (min(x.v, x.u) > min(y.v, y.u))
            return true;
        if (min(x.v, x.u) == min(y.v, y.u) && max(x.v, x.u) > max(y.v, y.u))
            return true;
    }
    return false;
}

bool operator==(edge const &x, edge const &y) {
    return x.u == y.u && x.v == y.v && x.w == y.w;
}
