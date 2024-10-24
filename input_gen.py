g = open("g_100.txt", "r+")

n = 100
m = 200

g.write(f"{n} {m}\n")
for i in range(1,n):
    g.write(f"{i} {i+1} 10\n")
g.write(f"{n} {1}\n")
for i in range(1,n+1):
    g.write(f"0 {i} 1\n")
g.close()
