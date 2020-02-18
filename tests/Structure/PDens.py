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
ag1, ag2 = universe.select_atom_pairs(selection1="name C3", selection2="name C4")
analysis = PDens(
    function=Magnitude(Distance(ag1=ag1, ag2=ag2, use_mic=False)),
    bounds=(1.0, 2.0),
    bin_width=0.01,
)
universe.add_analysis(analysis)
universe.run_analyses(stop=100)

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
