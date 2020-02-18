import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.structure import DProf, Position, Projection, Charge

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

ag = universe.select_atoms("not resname TIP3")
analysis = DProf(
    function=Projection(Position(ag=ag)),
    bounds=(-50.0, 50.0),
    bin_width=0.5,
    weight_function=Charge(ag=ag),
)
universe.add_analysis(analysis)

universe.run_analyses(stop=1000)

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
