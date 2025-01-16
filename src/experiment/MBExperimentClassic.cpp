#include <random>
#include <iostream>
#include <fstream>

#include "graph.h"
#include "mb/algo/MBClassicAlgorithm.h"
#include "mb/algo/MBFastAlgorithm.h"
#include "mb/MetricBackbone.h"


using namespace std;
typedef long long ll;


int main(int argc, char* argv[]) {

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " graph_filename" << '\n';
    }
    string filename = argv[1];

    cout << "reading graph: " << filename << '\n';

    graph G(filename);
    auto classic_algo = MBClassicAlgorithm(true);
    MBCalculationAlgorithm &alg_classic{classic_algo};

    MetricBackbone mb_classic = MetricBackbone(G, alg_classic);
    mb_classic.writeToFile("test_classic_mb.txt");
}
