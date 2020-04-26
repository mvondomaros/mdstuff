import numpy as np

from mdstuff import Universe
from mdstuff.distributions import Histogram2D
from mdstuff.structure import UserFunction, Dipoles
from mdstuff.transformations import Center


def mean_dihedrals(*selections):
    d = np.transpose([s.dihedrals() for s in selections])
    d *= 180.0 / np.pi
    return np.mean(d, axis=1)


u = Universe(
    topology="/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/Config/structure.psf",
    trajectories=[
        "/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/nvt.04.dcd",
        "/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/nvt.05.dcd",
    ],
)
t = Center()
u.add_transformations(t)


d1 = u.select_compounds("name C4", "name  C5", "name  C6", "name  C7")
d2 = u.select_compounds("name C9", "name  C10", "name  C11", "name  C12")
d3 = u.select_compounds("name C14", "name  C15", "name  C16", "name  C17")
d4 = u.select_compounds("name C18", "name  C20", "name  C21", "name  C22")
d5 = u.select_compounds("name C23", "name  C25", "name  C26", "name  C27")
dihedrals = d1 + d2 + d3 + d4 + d5

ch2 = u.select_compounds("type CTL2 HAL2")

a = Histogram2D(
    y_values=Dipoles(ch2, magnitude=True),
    y_bins=(0.0, 1.0, 0.005),
    x_values=UserFunction(mean_dihedrals, d1, d2, d3, d4, d5),
    x_bins=(0.0, 180.0, 1.0),
)

u.add_analyses(a)
u.run_analyses(step=2)

a.save(
    "Output/methylene_bridge_mean_dihedral_vs_colelctive_ch2_dipole_moment",
    normalization="pdens",
)
