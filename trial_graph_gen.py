import numpy as np
from EffectiveResistanceSampling.RanGraphGen import SBM


def output(adj, filename):
    with open(filename, "w+") as f:
        f.write(f"{len(adj)} {int(np.sum(adj) // 2)}\n")
        for i in range(len(adj)):
            for j in range(i + 1, len(adj[i])):
                if adj[i][j] == 1.0:
                    f.write(f"{i} {j} 1\n")


def modify(filename):
    l = open(filename, "r").readlines()
    e = list(map(lambda s: tuple(map(int, s.strip().split())), l))
    e_min = min(list(map(lambda edge: min(edge[0], edge[1]), e)))
    e_max = max(list(map(lambda edge: max(edge[0], edge[1]), e)))
    length = len(l)
    assert e_min == 1
    g = open(f"{filename}-uniform.txt", "w+")
    g.write(f"{e_max} {length}\n")
    for edge in e:
        g.write(f"{edge[0] - 1} {edge[1] - 1} {edge[2]}\n")
    g.close()


if __name__ == '__main__':
    modify("Datasets/USairport500.txt")
    # for n in (10, 20, 50, 100):
    #     adj = SBM((n // 2, n // 2), ((0.5, 0.1), (0.1, 0.5)))
    #     output(adj, f"Datasets/sbm/sbm_n={n}_p1=05,p2=01.txt")

