#include <random>
#include <iostream>

#include "graph.h"
#include "mb/algo/MBClassicAlgorithm.h"
#include "mb/algo/MBFastAlgorithm.h"
#include "mb/algo/MBFirstAndBFSAlgorithm.h"
#include "mb/algo/MBApproximationFordBellmanAlgorithm.h"
#include "mb/MetricBackbone.h"


using namespace std;
typedef long long ll;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " graph_filename" << '\n';
    }
    string graph_filename = argv[1];
    string filename_no_ext = graph_filename.substr(0, graph_filename.find_last_of('.'));
    cout << "reading graph: " << graph_filename << '\n';

    graph G(graph_filename);
    auto fb_algo = MBApproximationFordBellmanAlgorithm();
    MBCalculationAlgorithm &algorithm{fb_algo};

    MetricBackbone mb_original(G, algorithm);
    string mb_original_filename = filename_no_ext + "_mb.txt";
    mb_original.writeToFile(mb_original_filename);
}