import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.structure import PDens, Distance, Magnitude

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)
ag1, ag2 = universe.select_atom_pairs(
    selection1="type OT", selection2="type CEL1", mode="between"
)
analysis = PDens(
    function=Magnitude(Distance(ag1=ag1, ag2=ag2)),
    bounds=(0.0, 20.0),
    bin_width=0.1,
    mode="radial",
)
universe.add_analysis(analysis)
universe.run_analyses()

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
