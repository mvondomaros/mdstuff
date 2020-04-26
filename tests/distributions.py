import numpy as np

from mdstuff import Universe
from mdstuff.distributions import Histogram
from mdstuff.structure import CenterOfMass, Dipoles, Bonds, Dihedrals, UserFunction
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

molecules = u.select_compounds("all")
ch3 = u.select_compounds("type CTL3 HAL3")
ch2 = u.select_compounds("type CTL2 HAL2")
ch = u.select_compounds("type CEL1 HEL1")
ends = u.select_compounds("name C3", "name C28")

d1 = u.select_compounds("name C4", "name  C5", "name  C6", "name  C7")
d2 = u.select_compounds("name C9", "name  C10", "name  C11", "name  C12")
d3 = u.select_compounds("name C14", "name  C15", "name  C16", "name  C17")
d4 = u.select_compounds("name C18", "name  C20", "name  C21", "name  C22")
d5 = u.select_compounds("name C23", "name  C25", "name  C26", "name  C27")
dihedrals = d1 + d2 + d3 + d4 + d5

a1 = Histogram(values=Dipoles(molecules, magnitude=True), bins=(0.0, 1.0, 0.005),)
a2 = Histogram(values=Dipoles(ch3, magnitude=True), bins=(0.0, 1.0, 0.005),)
a3 = Histogram(values=Dipoles(ch2, magnitude=True), bins=(0.0, 1.0, 0.005),)
a4 = Histogram(values=Dipoles(ch, magnitude=True), bins=(0.0, 1.0, 0.005),)
a5 = Histogram(values=CenterOfMass(molecules, axis=2), bins=(-50.0, 50.0, 0.1),)
a6 = Histogram(values=Bonds(ends, distance=True), bins=(0.0, 30.0, 0.1),)
a7 = Histogram(values=Dihedrals(dihedrals), bins=(0.0, 180.0, 1.0))
a8 = Histogram(
    values=UserFunction(mean_dihedrals, d1, d2, d3, d4, d5), bins=(0.0, 180.0, 1.0)
)

u.add_analyses(a1, a2, a3, a4, a5, a6, a7)
u.run_analyses(step=2)

a1.save("Output/molecular_dipole_moment_distribution", normalization="pdens")
a2.save("Output/collective_ch3_dipole_moment_distribution", normalization="pdens")
a3.save("Output/collective_ch2_dipole_moment_distribution", normalization="pdens")
a4.save("Output/collective_ch_dipole_moment_distribution", normalization="pdens")
a5.save("Output/com_position_distribution", normalization="pdens")
a6.save("Output/e2e_distribution", normalization="pdens")
a7.save("Output/methylene_bridge_dihedral_distribution", normalization="pdens")
a8.save("Output/methylene_bridge_mean_dihedral_distribution", normalization="pdens")
