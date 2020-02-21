import glob
import os

import matplotlib.pyplot as plt

from mdstuff import NAMDUniverse
from mdstuff.dynamics import MSD

psf = os.path.abspath("../Data/OzoneSqualeneBulk/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/OzoneSqualeneBulk/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

# ag_list = universe.select_compounds("not resname SQU")
# analysis = MSD(ag_list)
# universe.add_analysis(analysis)
#
# universe.run_analyses()
#
# plt.figure()
# msd, time = analysis.get()
# plt.plot(time, msd)
# plt.show()
#
