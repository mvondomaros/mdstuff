import glob
import os
import warnings

import matplotlib.pyplot as plt

import mdstuff

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
analysis = mdstuff.structure.PDens2D(
    x_function=mdstuff.structure.InterCompoundDistance(
        selection="resname SQU", compound="residues"
    ),
    x_bounds=(0.0, 50.0),
    x_bin_width=1.0,
    y_function=mdstuff.structure.InterCompoundDistanceProjection(
        selection="resname SQU", compound="residues"
    ),
    y_bounds=(-50.0, 50.0),
    y_bin_width=1.0,
)
with warnings.catch_warnings():
    warnings.simplefilter("error")
    universe = mdstuff.NAMDUniverse(psf, dcds).add_analyses(analysis).run_analyses(stop=100)

plt.figure()
n, z1, z2 = analysis.get(centers=True)
plt.pcolormesh(z1, z2, n, shading="gouraud")
plt.colorbar()
plt.show()
plt.close()
