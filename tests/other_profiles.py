import numpy as np

from mdstuff import Universe
from mdstuff.distributions import Histogram2D
from mdstuff.structure import Bonds, CenterOfMass, UserFunction
from mdstuff.transformations import Center


def mean_dihedrals(*selections):
    d = np.transpose([s.dihedrals() for s in selections])
    d *= 180.0 / np.pi
    return np.mean(d, axis=1)


def dipole_tilt_angle(s):
    d = s.dipoles()
    cos_theta = d[:, 2] / np.linalg.norm(d, axis=1)
    gamma = np.arccos(cos_theta) * 180.0 / np.pi - 90
    return gamma


u = Universe(
    topology="/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/Config/structure.psf",
    trajectories=[
        "/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/nvt.04.dcd",
        "/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/nvt.05.dcd",
    ],
)
t = Center()
u.add_transformations(t)

molecules = u.select_compounds("all")
ends = u.select_compounds("name C3", "name C28")

d1 = u.select_compounds("name C4", "name  C5", "name  C6", "name  C7")
d2 = u.select_compounds("name C9", "name  C10", "name  C11", "name  C12")
d3 = u.select_compounds("name C14", "name  C15", "name  C16", "name  C17")
d4 = u.select_compounds("name C18", "name  C20", "name  C21", "name  C22")
d5 = u.select_compounds("name C23", "name  C25", "name  C26", "name  C27")
dihedrals = d1 + d2 + d3 + d4 + d5


a1 = Histogram2D(
    x_values=CenterOfMass(molecules, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Bonds(ends, distance=True),
    y_bins=(0.0, 30.0, 0.1),
)
a2 = Histogram2D(
    x_values=CenterOfMass(molecules, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=UserFunction(mean_dihedrals, d1, d2, d3, d4, d5),
    y_bins=(0.0, 180.0, 1.0),
)
a3 = Histogram2D(
    x_values=CenterOfMass(molecules, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=UserFunction(mean_dihedrals, d1, d2, d3, d4, d5),
    y_bins=(-90.0, 90.0, 1.0),
)

u.add_analyses(a1, a2, a3)
u.run_analyses(step=2)

a1.save("Output/e2e_profile", normalization="cond_pdens")
a2.save("Output/methylene_bridge_mean_dihedral_profile", normalization="cond_pdens")
a3.save("Output/collective_ch2_dipole_tilt_angle_profile", normalization="cond_pdens")
