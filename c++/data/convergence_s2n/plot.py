import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d as a3
import pandas
import numpy as np

masses = ["22", "23", "24"]

fig = plt.figure(figsize=(8.27, 11.69))

vtx1 = np.array([[300, 300, 10], [10,  300, 300], [10, 10,  10]])
vtx2 = np.array([[300, 300, 10], [300, 10 , 300], [10, 10,  10]])
vtx3 = np.array([[300,   10, 300], [10, 300, 300], [10, 10,  10]])

s2n_scale = 1e7
bs_scale = 1e4
def disable_ticks(ax):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)

def disable_xyticks(ax):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)

def disable_zticks(ax):
    ax.zaxis.set_ticklabels([])

    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)

for i, mass in enumerate(masses):
    data = pandas.read_csv("FDMFDM_"+mass+".dat", delim_whitespace=True, skiprows=[0])
    data.columns = ["l1", "l2", "l3", "B_k", "BKS", "Cov", "S2N"]

    ax = fig.add_subplot  (4, 2, 2*i + 1, projection='3d')
    surf = ax.scatter(data["l1"],data["l2"], data["l3"], s=data["BKS"]*s2n_scale, c=data["BKS"]*s2n_scale, cmap=cm.coolwarm)
    tri = a3.art3d.Poly3DCollection([vtx1])
    tri.set_color("white")
    tri.set_edgecolor('gray')
    ax.add_collection3d(tri)
    ax.set_xlabel("$l_1$")
    ax.set_ylabel("$l_2$")
    ax.set_zlabel("$l_3$")
    #disable_ticks(ax)

    ax = fig.add_subplot  (4, 2, 2*i + 2, projection='3d')
    surf = ax.scatter(data["l1"],data["l2"], data["l3"], s=data["S2N"]*bs_scale, c=data["S2N"]*bs_scale, cmap=cm.coolwarm)
    tri = a3.art3d.Poly3DCollection([vtx1])
    tri.set_color("white")
    tri.set_edgecolor('gray')
    ax.add_collection3d(tri)
    ax.set_xlabel("$l_1$")
    ax.set_ylabel("$l_2$")
    ax.set_zlabel("$l_3$")
    #disable_xyticks(ax)

data = pandas.read_csv("CDM_gaussian.dat", delim_whitespace=True, skiprows=[0])
data.columns = ["l1", "l2", "l3", "B_k", "BKS", "Cov", "S2N"]

ax = fig.add_subplot  (4, 2, 7, projection='3d')
ax.set_xlabel("$l_1$")
ax.set_ylabel("$l_2$")
ax.set_zlabel("$l_3$")
#disable_zticks(ax)
surf = ax.scatter(data["l1"],data["l2"], data["l3"], s=data["BKS"]*s2n_scale, c=data["BKS"]*s2n_scale, cmap=cm.coolwarm)
tri = a3.art3d.Poly3DCollection([vtx1])
tri.set_color("white")
tri.set_edgecolor('black')
ax.add_collection3d(tri)

ax = fig.add_subplot  (4, 2, 8, projection='3d')
ax.set_xlabel("$l_1$")
ax.set_ylabel("$l_2$")
ax.set_zlabel("$l_3$")
surf = ax.scatter(data["l1"],data["l2"], data["l3"], s=data["S2N"]*bs_scale, c=data["S2N"]*bs_scale, cmap=cm.coolwarm)
tri = a3.art3d.Poly3DCollection([vtx1])
tri.set_color("white")
tri.set_edgecolor('black')
ax.add_collection3d(tri)

#fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.00, wspace=0, hspace=0.03)
plt.savefig("Lensing.eps", format="eps", dpi = 1200)
plt.clf()
