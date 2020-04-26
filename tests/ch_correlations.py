import numpy as np

from mdstuff import Universe
from mdstuff.distributions import Histogram
from mdstuff.structure import UserFunction
from mdstuff.transformations import Center


def bond_bond_correlations(b1, b2):
    d1 = b1.bonds()
    d2 = b2.bonds()
    d1_norm = np.linalg.norm(d1, axis=1)
    d2_norm = np.linalg.norm(d2, axis=1)
    return np.sum(d1 * d2, axis=1) / (d1_norm * d2_norm)


u = Universe(
    topology="/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/Config/structure.psf",
    trajectories=[
        "/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/nvt.04.dcd",
        "/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/nvt.05.dcd",
    ],
)
t = Center()
u.add_transformations(t)

ch2 = u.select_compounds("name C4", "name H4")
ch6 = u.select_compounds("name C9", "name H9")
ch10 = u.select_compounds("name C14", "name H14")
ch13 = u.select_compounds("name C17", "name H17")
ch17 = u.select_compounds("name C22", "name H221")
ch21 = u.select_compounds("name C27", "name H27")

a_2_6 = Histogram(
    values=UserFunction(bond_bond_correlations, ch2, ch6), bins=(-1.0, 1.0, 0.01),
)
a_2_10 = Histogram(
    values=UserFunction(bond_bond_correlations, ch2, ch10), bins=(-1.0, 1.0, 0.01),
)
a_2_13 = Histogram(
    values=UserFunction(bond_bond_correlations, ch2, ch13), bins=(-1.0, 1.0, 0.01),
)
a_2_17 = Histogram(
    values=UserFunction(bond_bond_correlations, ch2, ch17), bins=(-1.0, 1.0, 0.01),
)
a_2_21 = Histogram(
    values=UserFunction(bond_bond_correlations, ch2, ch21), bins=(-1.0, 1.0, 0.01),
)

a_6_10 = Histogram(
    values=UserFunction(bond_bond_correlations, ch6, ch10), bins=(-1.0, 1.0, 0.01),
)
a_6_13 = Histogram(
    values=UserFunction(bond_bond_correlations, ch6, ch13), bins=(-1.0, 1.0, 0.01),
)
a_6_17 = Histogram(
    values=UserFunction(bond_bond_correlations, ch6, ch17), bins=(-1.0, 1.0, 0.01),
)
a_6_21 = Histogram(
    values=UserFunction(bond_bond_correlations, ch6, ch21), bins=(-1.0, 1.0, 0.01),
)

a_10_13 = Histogram(
    values=UserFunction(bond_bond_correlations, ch10, ch13), bins=(-1.0, 1.0, 0.01),
)
a_10_17 = Histogram(
    values=UserFunction(bond_bond_correlations, ch10, ch17), bins=(-1.0, 1.0, 0.01),
)
a_10_21 = Histogram(
    values=UserFunction(bond_bond_correlations, ch10, ch21), bins=(-1.0, 1.0, 0.01),
)

a_13_17 = Histogram(
    values=UserFunction(bond_bond_correlations, ch13, ch17), bins=(-1.0, 1.0, 0.01),
)
a_13_21 = Histogram(
    values=UserFunction(bond_bond_correlations, ch13, ch21), bins=(-1.0, 1.0, 0.01),
)

a_17_21 = Histogram(
    values=UserFunction(bond_bond_correlations, ch17, ch21), bins=(-1.0, 1.0, 0.01),
)

u.add_analyses(
    a_2_6,
    a_2_10,
    a_2_13,
    a_2_17,
    a_2_21,
    a_6_10,
    a_6_13,
    a_6_17,
    a_6_21,
    a_10_13,
    a_10_17,
    a_10_21,
    a_13_17,
    a_13_21,
    a_17_21,
)
u.run_analyses(step=2)

a_2_6.save("Output/ch_correlations_2_6", normalization="pdens")
a_2_10.save("Output/ch_correlations_2_10", normalization="pdens")
a_2_13.save("Output/ch_correlations_2_13", normalization="pdens")
a_2_17.save("Output/ch_correlations_2_17", normalization="pdens")
a_2_21.save("Output/ch_correlations_2_21", normalization="pdens")

a_6_10.save("Output/ch_correlations_6_10", normalization="pdens")
a_6_13.save("Output/ch_correlations_6_13", normalization="pdens")
a_6_17.save("Output/ch_correlations_6_17", normalization="pdens")
a_6_21.save("Output/ch_correlations_6_21", normalization="pdens")

a_10_13.save("Output/ch_correlations_10_13", normalization="pdens")
a_10_17.save("Output/ch_correlations_10_17", normalization="pdens")
a_10_21.save("Output/ch_correlations_10_21", normalization="pdens")

a_13_17.save("Output/ch_correlations_13_17", normalization="pdens")
a_13_21.save("Output/ch_correlations_13_21", normalization="pdens")

a_17_21.save("Output/ch_correlations_17_21", normalization="pdens")
