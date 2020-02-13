import glob
import os

import matplotlib.pyplot as plt

import mdstuff

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p)
    for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
analysis = mdstuff.structure.Prof(
    function=mdstuff.structure.CompoundAxisPosition(
        selection="resname TIP3", compound="residues"
    ),
    bounds=(0.0, 50.0),
    bin_width=1.0,
    weight_function=mdstuff.structure.DipoleOrientation(
        "resname TIP3", compound="residues"
    ),
)
universe = mdstuff.NAMDUniverse(psf, dcds).add_analyses(analysis).run_analyses()

plt.figure()
n, z = analysis.get(centers=True)
plt.plot(z, n)
plt.show()
plt.close()
