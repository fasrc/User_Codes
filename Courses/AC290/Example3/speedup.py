"""
Program: speedup.py
         Code generates speedup plot
         for Number of MPI tasks = [1, 2, 4 ,8, 16]
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17

nproc      = [1, 2, 4, 8, 16]
walltime   = [12.27, 6.19, 3.43, 1.84, 0.90]

speedup = []
efficiency = []
n = range(5)
for i in n:
    s = walltime[0] / walltime[i]
    e = 100 * s / (2**i)
    speedup.append(s)
    efficiency.append(e)

# Print out results
print " # MPI tasks  Walltime  Speedup  Efficiency (%)"
for i in n:
    print "%8d %11.2f %8.2f %11.2f" % \
        (nproc[i], walltime[i], speedup[i], efficiency[i])
    

fig, ax = plt.subplots(figsize=(8,6))
p1 = plt.plot(nproc, nproc, linewidth = 2.0, color="black",
        linestyle='-', label='Ideal speedup')
p2 = plt.plot(nproc, speedup, linewidth = 2.0, color="red",
        linestyle='--', label='Speedup')
plt.xlabel('Number of MPI tasks', fontsize=20)
plt.ylabel('Speedup', fontsize=20)
plt.legend(fontsize=15,loc=2)
plt.xlim(1, 16)
plt.ylim(1, 16)

plt.savefig('speedup.png', format='png')
plt.show()
