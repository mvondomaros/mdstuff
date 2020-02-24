import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.structure import Prof, Position, Projection, Mass

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

ag = universe.select_atoms("not resname TIP3")
analysis = Prof(
    x_function=Projection(Position(ag=ag)),
    x_bounds=(-50.0, 50.0),
    x_bin_width=0.5,
    y_function=Mass(ag=ag),
)
universe.add_analysis(analysis)

universe.run_analyses(stop=100)

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
