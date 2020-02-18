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

# Selection mode: zip.
ag1, ag2 = universe.select_atom_pairs(selection1="name C3", selection2="name C4")
analysis1 = PDens(
    function=Magnitude(Distance(ag1=ag1, ag2=ag2, use_mic=False)),
    bounds=(1.0, 2.0),
    bin_width=0.01,
)
universe.add_analysis(analysis1)

# Selection mode: within.
ag1, ag2 = universe.select_atom_pairs(
    selection1="name C3", selection2="name C4", mode="within"
)
analysis2 = PDens(
    function=Magnitude(Distance(ag1=ag1, ag2=ag2, use_mic=False)),
    bounds=(1.0, 2.0),
    bin_width=0.01,
)
universe.add_analysis(analysis2)

universe.run_analyses(stop=100)

plt.figure()
n, z = analysis1.get(centers=True)
plt.plot(z, n)
n, z = analysis2.get(centers=True)
plt.plot(z, n)
plt.show()
