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

ch_1 = u.select_compounds("name C4 H4")
ch_2 = u.select_compounds("name C9 H9")
ch_3 = u.select_compounds("name C14 H14")
ch_4 = u.select_compounds("name C17 H17")
ch_5 = u.select_compounds("name C22 H221")
ch_6 = u.select_compounds("name C27 H27")
ch = ch_1 + ch_2 + ch_3 + ch_4 + ch_5 + ch_6

ch_a = Histogram2D(
    x_values=CenterOfMass(ch, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Dipoles(ch, axis=2, orientation=True),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(ch_a)


ch2_1 = u.select_compounds("name C5 H51 H52")
ch2_2 = u.select_compounds("name C6 H61 H62")
ch2_3 = u.select_compounds("name C10 H101 H102")
ch2_4 = u.select_compounds("name C11 H111 H112")
ch2_5 = u.select_compounds("name C15 H151 H152")
ch2_6 = u.select_compounds("name C16 H161 H162")
ch2_7 = u.select_compounds("name C20 H201 H202")
ch2_8 = u.select_compounds("name C21 H211 H212")
ch2_9 = u.select_compounds("name C25 H251 H252")
ch2_10 = u.select_compounds("name C26 H261 H262")
ch2 = ch2_1 + ch2_2 + ch2_3 + ch2_4 + ch2_5 + ch2_6 + ch2_7 + ch2_8 + ch2_9 + ch2_10

ch2_a = Histogram2D(
    x_values=CenterOfMass(ch2, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Dipoles(ch2, axis=2, orientation=True),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(ch2_a)


ch3_1 = u.select_compounds("name C1 H11 H12 H13")
ch3_2 = u.select_compounds("name C2 H21 H22 H23")
ch3_3 = u.select_compounds("name C8 H81 H82 H83")
ch3_4 = u.select_compounds("name C13 H131 H132 H133")
ch3_5 = u.select_compounds("name C19 H191 H192 H193")
ch3_6 = u.select_compounds("name C24 H241 H242 H243")
ch3_7 = u.select_compounds("name C29 H291 H292 H293")
ch3_8 = u.select_compounds("name C30 H301 H302 H303")
ch3 = ch3_1 + ch3_2 + ch3_3 + ch3_4 + ch3_5 + ch3_6 + ch3_7 + ch3_8

ch3_a = Histogram2D(
    x_values=CenterOfMass(ch3, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Dipoles(ch3, axis=2, orientation=True),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(ch3_a)

db_1 = u.select_compounds("name C3 C4")
db_2 = u.select_compounds("name C7 C9")
db_3 = u.select_compounds("name C12 C14")
db_4 = u.select_compounds("name C18 C17")
db_5 = u.select_compounds("name C23 C22")
db_6 = u.select_compounds("name C28 C27")
db = db_1 + db_2 + db_3 + db_4 + db_5 + db_6

db_a = Histogram2D(
    x_values=CenterOfMass(db, axis=2),
    x_bins=(-40.0, 40.0, 0.1),
    y_values=Dipoles(db, axis=2, orientation=True),
    y_bins=(-1.0, 1.0, 0.01),
)
u.add_analyses(db_a)


u.run_analyses(step=2)

ch_a.save("Output/individual_ch_orientation_profile", normalization="cond_pdens")
ch2_a.save("Output/individual_ch2_orientation_profile", normalization="cond_pdens")
ch3_a.save("Output/individual_ch3_orientation_profile", normalization="cond_pdens")
db_a.save("Output/individual_db_orientation_profile", normalization="cond_pdens")
