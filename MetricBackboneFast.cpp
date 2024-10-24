#include <cassert>
#include <vector>
#include <unordered_map>
#include <random>
#include <iostream>
#include <fstream>
#include <queue>
#include <set>
#include <unordered_set>


using namespace std;
typedef long long ll;

struct pair_hash {
    inline std::size_t operator()(const std::pair<int, int> &v) const {
        return v.first * 31 + v.second;
    }
};

struct edge {
    int u, v;
    double w;

    edge() {
    }

    edge(int _u, int _v, double _w) : u(_u), v(_v), w(_w) {
    }

    friend bool operator<(edge const &x, edge const &y) {
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

    friend bool operator>(edge const &x, edge const &y) {
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
};

struct node {
    vector<edge> edges;
};

struct graph {
    int n, m;
    string path;
    vector<node> nodes;

    unordered_map<pair<int, int>, double, pair_hash> edgesMB;

    graph(const string &_path) : path(_path) {
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

    void add_edge(int u, int v, double w) {
        nodes[u].edges.emplace_back(u, v, w);
        nodes[v].edges.emplace_back(v, u, w);
    }

    void getMB(bool approx = false) {
        vector<int> nodesToTry;
        for (int i = 0; i < n; i++) { nodesToTry.push_back(i); }
        if (approx) {
            int cnt = 2 * log(n) + 1;
            auto rd = std::random_device{};
            auto rng = std::default_random_engine{rd()};
            shuffle(nodesToTry.begin(), nodesToTry.end(), rng);
            nodesToTry.erase(nodesToTry.begin() + cnt, nodesToTry.end());
        }

        for (auto s: nodesToTry) {
            vector<pair<int, double> > parent(n, {-1, -1});
            getSPT(s, parent);
            for (int u = 0; u < n; u++) {
                auto [v, w] = parent[u];
                if (v == -1) { continue; }
                edgesMB[{u, v}] = w;
            }
        }
    }

    set<edge> getMBFast() {
        vector<edge> edges;
        edges.reserve(m);
        for (int i = 0; i < n; i++) {
            for (auto &e: nodes[i].edges) {
                if (e.u < e.v) {
                    // we don't allow 0-loops
                    edges.emplace_back(e.u, e.v, e.w);
                }
            }
        }
        vector<edge> edges1;
        for (int i = 0; i < m; i++) {
            auto e = edges[i];
            bool take = true;
            int u = e.u, v = e.v;
            for (auto &x: nodes[u].edges) {
                for (auto &y: nodes[v].edges) {
                    if (x.v == y.v && x.w + y.w < e.w) {
                        take = false;
                        goto loop_end;
                    }
                }
            }
        loop_end:
            if (take) {
                edges1.emplace_back(u, v, e.w);
            }
        }

        cout << "possibly metric edges after step 1: size " << edges1.size() << '\n';
        for (auto &e: edges1) {
            if (e.u == 169) {
                cout << e.u << " " << e.v << " " << e.w << endl;
            }
        }
        cout << '\n';

        vector<priority_queue<edge, vector<edge>, greater<> > > vi(n);
        // fill the ordered nodes adjacency lists
        for (auto &e: edges1) {
            vi[e.u].emplace(e.u, e.v, e.w);
            vi[e.v].emplace(e.v, e.u, e.w);
        }
        set<edge> ms;
        for (int i = 0; i < n; i++) {
            set<edge> ms_local;
            if (!vi[i].empty()) {
                auto least_w = vi[i].top().w;
                while (!vi[i].empty()) {
                    if (least_w == vi[i].top().w) {
                        ms_local.insert(vi[i].top());
                        vi[i].pop();
                        continue;
                    }
                    edge e = vi[i].top();
                    vi[i].pop();
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

        cout << "metric edges after step 2: size " << ms.size() << '\n';
        // for (auto &e: ms) {
        //     cout << e.u << " " << e.v << " " << e.w << endl;
        // }
        // cout << '\n';

        set<edge> u;
        set_difference(edges1.begin(), edges1.end(), ms.begin(), ms.end(), std::inserter(u, u.end()));

        for (auto &e: u) {
            if (getMetricBFS(e)) {
                ms.insert(e);
            }
        }
        return ms;
    }

    bool getMetricBFS(const edge &e) {
        int u = e.u, v = e.v;
        int d = e.w;
        vector dist(n, 1e15);
        vector visited(n, false);
        priority_queue<pair<ll, int>, vector<pair<ll, int> >, greater<> > pq;
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

            for (auto &x: nodes[cur].edges)
                if (dist[x.v] > dist[cur] + x.w) {
                    dist[x.v] = dist[cur] + x.w;
                    pq.emplace(dist[x.v], x.v);
                }
        }
        return true;
    }

    void getSPT(int s, vector<pair<int, double> > &parent) {
        vector<double> dist(n, 1e15);
        vector<bool> visited(n, false);
        priority_queue<pair<ll, int>, vector<pair<ll, int> >, greater<> > pq;
        dist[s] = 0ll;
        pq.emplace(0, s);

        while (!pq.empty()) {
            int cur = pq.top().second;
            pq.pop();
            if (visited[cur]) { continue; }
            visited[cur] = true;

            for (auto &e: nodes[cur].edges)
                if (dist[e.v] > dist[cur] + e.w) {
                    dist[e.v] = dist[cur] + e.w;
                    parent[e.v] = {e.u, e.w};
                    pq.emplace(dist[e.v], e.v);
                }
        }
    }

    void writeMBToFile(bool approx = false) {
        string curPath = "output/MB_" + path;
        if (approx) { curPath = "output/MBApprox_" + path; }
        ofstream ofs(curPath);
        ofs << n << '\n';
        for (auto &[p, w]: edgesMB) {
            if (p.first < p.second) {
                ofs << p.first << " " << p.second << " " << w << '\n';
            }
        }
        ofs.close();
    }

    void check() {
        vector<vector<double> > dist2(n, vector<double>(n, 1e8));
        for (int i = 0; i < n; i++) {
            dist2[i][i] = 0;
            for (auto &e: nodes[i].edges) {
                dist2[e.u][e.v] = e.w;
            }
        }
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    dist2[i][j] = min(dist2[i][j], dist2[i][k] + dist2[j][k]);
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (auto &e: nodes[i].edges) {
                if (e.w == dist2[e.u][e.v]) {
                    assert(edgesMB.count({e.u, e.v}));
                } else {
                    assert(!edgesMB.contains({e.u, e.v}));
                }
            }
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr), cout.tie(nullptr);

    // bool approx = true;

    // graph G("Datasets/openflights-uniform.txt");
    graph G("Datasets/USairport500-uniform.txt");
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // graph G("Cifar_122.txt");
    // G.getMB(false);
    // G.writeMBToFile(approx);

    set<edge> metric = G.getMBFast();
    //
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //
    std::cout << "Time taken to build MB new method: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).
            count() <<
            " s" << std::endl;

    // cout << "number of metric edges: " << metric.size() << '\n';

    // G.getMB(false);

    // begin = std::chrono::steady_clock::now();

    // end = std::chrono::steady_clock::now();
    // std::cout << "Time taken to build MB old method: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() <<
    //     " s" << std::endl;
    //
    // G.writeMBToFile(false);

    // graph G("Datasets/sbm/sbm_n=10_p1=05,p2=01.txt");


    // ofstream ofs("output/MB_Datasets/USairport500-uniform-fast.txt");
    // ofs << G.n << " " << metric.size() << std::endl;
    // for (auto &e: metric) {
    //     ofs << e.u << " " << e.v << " " << e.w << '\n';
    // }
    // ofs.close();
}
