//
// Created by Oleg Fafurin on 13.11.2024.
//

#ifndef GRAPH_H
#define GRAPH_H

#include <set>
#include <vector>
#include <unordered_map>
typedef long long ll;


struct edge {
    int u, v;
    double w;

    edge();

    edge(int _u, int _v, double _w);

    friend bool operator<(edge const &x, edge const &y);

    friend bool operator>(edge const &x, edge const &y);

    friend bool operator==(edge const &x, edge const &y);
};

struct node {
    std::vector<edge> edges;
};

struct pair_hash {
    std::size_t operator()(const std::pair<int, int> &v) const {
        return v.first * 31 + v.second;
    }
};

struct graph {
    int n, m;
    std::string path;
    std::vector<node> nodes;

    explicit graph(const std::string &_path);

    graph(const graph &g);

    graph(int n);

    // void getMB(int n_roots);

    void getSPT(int s, std::vector<std::pair<int, double> > &parent, const std::vector<node> &vertexEdges) const;

    void writeToFile(const std::string &filename, bool include_header);

private:
    void add_edge(int u, int v, double w);
};

#endif //GRAPH_H
