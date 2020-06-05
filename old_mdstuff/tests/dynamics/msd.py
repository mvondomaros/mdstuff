import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.dynamics import SingleParticleMSD

psf = os.path.abspath("../data/OzoneSqualeneBulk/structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../data/OzoneSqualeneBulk/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

ag = universe.select_atoms("not resname SQU")
analysis = SingleParticleMSD(ag)
universe.add_analysis(analysis)

universe.run_analyses()

plt.figure()
msd, time = analysis.get()
plt.plot(time, msd)

plt.show()



