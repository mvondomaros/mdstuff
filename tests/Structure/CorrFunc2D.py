import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.structure import CorrFunc2D, Distance, Magnitude

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

ag1, ag2 = universe.select_atom_pairs(
    selection1="name C3", selection2="name C4", mode="within"
)
ag3, ag4 = universe.select_atom_pairs(
    selection1="name C3", selection2="name C4", mode="within"
)
analysis = CorrFunc2D(
    x_function=Magnitude(Distance(ag1=ag1, ag2=ag2, use_mic=False)),
    x_bounds=(1.2, 1.8),
    x_bin_width=0.01,
    y_function=Magnitude(Distance(ag1=ag3, ag2=ag4, use_mic=False)),
    y_bounds=(1.2, 1.5),
    y_bin_width=0.01,
)
universe.add_analysis(analysis)

universe.run_analyses(stop=100)

plt.figure()
n, x, y = analysis.get(centers=True)
plt.pcolormesh(x, y, n, shading="gouraud")
plt.colorbar()
plt.show()
