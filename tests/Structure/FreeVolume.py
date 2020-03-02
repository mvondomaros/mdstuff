import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.structure import FreeVolumeProfile

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

ag = universe.select_atoms("resname SQU")
analysis = FreeVolumeProfile(ag=ag, bin_width=0.5)
universe.add_analysis(analysis)
universe.run_analyses(stop=1)

plt.figure()
n, z = analysis.get()
plt.plot(z, n)
plt.show()
