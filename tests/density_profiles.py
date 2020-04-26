from mdstuff import Universe
from mdstuff.distributions import Histogram
from mdstuff.structure import Masses, Positions, Charges
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
ch = u.select_compounds("type CEL1 HEL1")
ch2 = u.select_compounds("type CTL2 HAL2")
ch3 = u.select_compounds("type CTL3 HAL3")

a_mass = Histogram(
    values=Positions(squalene, axis=2),
    bins=(-50.0, 50.0, 0.5),
    weights=Masses(squalene),
)
a_charge = Histogram(
    values=Positions(squalene, axis=2),
    bins=(-50.0, 50.0, 0.5),
    weights=Charges(squalene),
)
a_charge_ch = Histogram(
    values=Positions(ch, axis=2), bins=(-50.0, 50.0, 0.5), weights=Charges(ch)
)
a_charge_ch2 = Histogram(
    values=Positions(ch2, axis=2), bins=(-50.0, 50.0, 0.5), weights=Charges(ch2)
)
a_charge_ch3 = Histogram(
    values=Positions(ch3, axis=2), bins=(-50.0, 50.0, 0.5), weights=Charges(ch3)
)

u.add_analyses(a_mass, a_charge, a_charge_ch, a_charge_ch3, a_charge_ch2)
u.run_analyses(step=2)

a_mass.save("Output/mass_density_profile", normalization="ave")
a_charge.save("Output/charge_density_profile", normalization="ave")
a_charge_ch.save("Output/ch_charge_density_profile", normalization="ave")
a_charge_ch2.save("Output/ch2_charge_density_profile", normalization="ave")
a_charge_ch3.save("Output/ch3_charge_density_profile", normalization="ave")
