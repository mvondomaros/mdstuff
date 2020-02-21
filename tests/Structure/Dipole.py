import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.structure import Prof, Dipole, Orientation, CompoundDistance, Projection

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)


analysis = Prof(
    function=Projection(
        CompoundDistance(
            universe=universe,
            selection1="resname TIP3",
            selection2="not resname TIP3",
            compound2="segments",
        )
    ),
    bounds=(-50.0, 50.0)
    bin_width=1.0,
    weight_function=Orientation(Dipole(universe=universe, selection="resname TIP3")),
)
universe.add_analysis(analysis)

universe.run_analyses()

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
