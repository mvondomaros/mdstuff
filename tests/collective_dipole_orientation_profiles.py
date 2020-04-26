from mdstuff import Universe
from mdstuff.distributions import Histogram2D
from mdstuff.structure import Dipoles, CenterOfMass
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


ch = u.select_compounds("type CEL1 HEL1")

a_ch = Histogram2D(
    x_values=CenterOfMass(ch, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Dipoles(ch, orientation=True, axis=2),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(a_ch)


ch2 = u.select_compounds("type CTL2 HAL2")

a_ch2 = Histogram2D(
    x_values=CenterOfMass(ch2, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Dipoles(ch2, orientation=True, axis=2),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(a_ch2)


ch3 = u.select_compounds("type CTL3 HAL3")

a_ch3 = Histogram2D(
    x_values=CenterOfMass(ch3, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Dipoles(ch3, orientation=True, axis=2),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(a_ch3)


u.run_analyses(step=2)

a_ch.save("Output/collective_ch_dipole_orientation_profile", normalization="cond_pdens")
a_ch2.save(
    "Output/collective_ch2_dipole_orientation_profile", normalization="cond_pdens"
)
a_ch3.save(
    "Output/collective_c3_dipole_orientation_profile", normalization="cond_pdens"
)
