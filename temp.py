


def compare(f1, f2):
    s1 = read_sorted(f1)
    s2 = read_sorted(f2)
    print(f"f1 - f2:\n{s1.difference(s2)}\n\n")
    print(f"f2 - f1:\n{s2.difference(s1)}\n\n")



def read_sorted(filename):
    with open(filename, "r") as f:
        lines = list(map(lambda line: list(map(int, line.strip().split())), f.readlines()[1:]))
        return {(min(line[0], line[1]), max(line[0], line[1]), line[2]) for line in lines}


if __name__ == '__main__':
    compare("output/MB_Datasets/USairport500-uniform.txt", "output/MB_Datasets/USairport500-uniform-fast.txt")