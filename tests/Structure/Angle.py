import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.structure import CorrFunc2D, PDens2D, Distance, Magnitude, Angle

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

ag1, ag2 = universe.select_atom_pairs(
    selection1="type OT", selection2="type CEL1", mode="product"
)
ag1, ag2, ag3 = universe.select_atom_triplets(
    ag1=ag1, ag2=ag2, selection="type HT and name H1", mode="product"
)

analysis = CorrFunc2D(
    x_function=Magnitude(Distance(ag1=ag1, ag2=ag2)),
    x_bounds=(0.0, 10.0),
    x_bin_width=0.5,
    x_mode="radial",
    y_function=Angle(vertex=ag1, tip1=ag2, tip2=ag3),
    y_bounds=(0.0, 180.0),
    y_bin_width=5.0,
    y_mode="angular",
)
universe.add_analysis(analysis)

universe.run_analyses(stop=10000)

plt.figure()
n, x, y = analysis.get(centers=True)
plt.pcolormesh(x, y, n, shading="gouraud")
plt.colorbar()
plt.show()
