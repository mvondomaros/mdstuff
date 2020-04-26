from mdstuff import Universe
from mdstuff.distributions import Histogram2D
from mdstuff.structure import Bonds, CenterOfMass, PrincipalAxes, Dipoles
from mdstuff.transformations import Center

u = Universe(
    topology="/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/Config/structure.psf",
    trajectories=[
        "/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/nvt.04.dcd",
        "/home/michael/Work/IndoorAir/Skin/Squalene/Slab/Analyses/Trajectories/nvt.05.dcd",
    ],
)
t = Center()
u.add_transformations(t)

squalene = u.select_compounds("all")

a_ax3 = Histogram2D(
    x_values=CenterOfMass(squalene, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=PrincipalAxes(squalene, n=3, orientation=2),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(a_ax3)

a_dipole = Histogram2D(
    x_values=CenterOfMass(squalene, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Dipoles(squalene, axis=2, orientation=True),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(a_dipole)


ends = u.select_compounds("name C3", "name C28")

a_e2e = Histogram2D(
    x_values=CenterOfMass(squalene, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Bonds(ends, orientation=2),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(a_e2e)


u.run_analyses(step=2)

a_ax3.save("Output/ax3_orientation_profile", normalization="cond_pdens")
a_dipole.save("Output/molecular_dipole_orientation_profile", normalization="cond_pdens")
a_e2e.save("Output/e2e_orientation_profile", normalization="cond_pdens")
