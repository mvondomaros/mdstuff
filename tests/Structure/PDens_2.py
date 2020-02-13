import glob
import os

import matplotlib.pyplot as plt

import mdstuff

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
analysis = mdstuff.structure.PDens(
    function=mdstuff.structure.CompoundAxisPosition(
        selection="resname TIP3", compound="group"
    ),
    bounds=(0.0, 50.0),
    bin_width=1.0,
)
universe = mdstuff.NAMDUniverse(psf, dcds).add_analyses(analysis).run_analyses()

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
plt.close()
