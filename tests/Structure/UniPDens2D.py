import glob
import os

import matplotlib.pyplot as plt

import mdstuff

psf = os.path.abspath("../Data/WaterSqualeneSlab/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/WaterSqualeneSlab/*.dcd"))
]
universe = mdstuff.NAMDUniverse(psf, dcds)


n, z1, z2 = (
    mdstuff.structure.UniPDens2D(
        x_function=mdstuff.structure.CompoundDistanceProjection(
            universe=universe, selection="name C3 C4", compound="residues",
        ),
        x_bounds=(0.0, 50.0),
        x_bin_width=0.5,
        y_function=mdstuff.structure.IntraCompoundDistance(
            universe=universe,
            selection1="name C3",
            selection2="name C4",
            compound="residues",
        ),
        y_bounds=(1.2, 1.5),
        y_bin_width=0.01,
    )
    .run()
    .get(centers=True)
)

plt.figure()
plt.pcolormesh(z1, z2, n, shading="gouraud")
plt.colorbar()
plt.show()
plt.close()
