import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.structure import (
    Prof,
    Dipole,
    Magnitude,
    CompoundDistance,
    Projection,
)

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

ag_list1 = universe.select_compounds("resname TIP3")
ag_list2 = universe.select_compounds("not resname TIP3", compound="group")

analysis = Prof(
    x_function=Projection(CompoundDistance(ag_list1=ag_list1, ag_list2=ag_list2,)),
    x_bounds=(-50.0, 50.0),
    x_bin_width=1.0,
    y_function=Magnitude(Dipole(ag_list=ag_list1)),
)
universe.add_analysis(analysis)

universe.run_analyses(stop=100)

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
