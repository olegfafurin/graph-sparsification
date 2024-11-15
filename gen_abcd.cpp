//
// Created by Oleg Fafurin on 14.11.2024.
//

#include <vector>
#include <random>
#include <iostream>
#include <fstream>

#include "graph.h"

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr), cout.tie(nullptr);


    for (auto x: {1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000}) {
        for (int iter_n = 1; iter_n <= 5; iter_n++) {
            string filename = "Datasets/abcd/n=" + to_string(x) + "/edge.txt";
            graph G(filename);
            int n = G.n;
            vector<pair<int, double> > parent(n, {-1, -1});

            int n_roots = 4 * log(x);

            vector<int> roots;
            roots.reserve(n);
            for (int i = 0; i < n; i++) { roots.push_back(i); }

            auto rd = std::random_device{};
            auto rng = std::default_random_engine{rd()};
            shuffle(roots.begin(), roots.end(), rng);
            roots.erase(roots.begin() + n_roots, roots.end());
            vector<long long> times(n_roots);
            for (int i = 1; i <= n_roots; ++i) {
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                // G.getMB(false);
                G.addSPT(roots[i]);
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
                times[i - 1] = time;
                G.writeMBToFile(
                    "Datasets/abcd/n=" + to_string(x) + "/approx/" + to_string(iter_n) + "/edges_" + to_string(i) +
                    "_SPTs.txt");
            }
            long long total_time = 0;
            for (int i = 0; i < n_roots; ++i) {
                total_time += times[i];
            }
            long long avg_time = total_time / n_roots;
            cout << "n=" << n << "; m=" << G.m << "; #SPTs: " << n_roots << "; total time:" << total_time <<
                    "ms; avg time per spt: " << avg_time << "ms" << '\n';
        }
    }
}
