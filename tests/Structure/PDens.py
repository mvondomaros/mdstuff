import glob
import os

import matplotlib.pyplot as plt

import mdstuff

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p)
    for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
analysis = mdstuff.structure.PDens(
    function=mdstuff.structure.IntraCompoundDistance(
        selection1="name C3", selection2="name C4", compound="residues"
    ),
    bounds=(1.0, 1.8),
    bin_width=0.01,
)
universe = mdstuff.NAMDUniverse(psf, dcds).add_analyses(analysis).run_analyses()

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
plt.close()
