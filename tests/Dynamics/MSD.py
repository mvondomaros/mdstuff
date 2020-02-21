import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from mdstuff import NAMDUniverse
from mdstuff.dynamics import SingleParticleMSD

psf = os.path.abspath("../Data/OzoneSqualeneBulk/Structure.psf")
dcds = [
    os.path.abspath(p) for p in sorted(glob.glob("../Data/OzoneSqualeneBulk/*.dcd"))
]
universe = NAMDUniverse(psf, dcds)

ag = universe.select_atoms("not resname SQU")
analysis = SingleParticleMSD(ag, n=25)
universe.add_analysis(analysis)

universe.run_analyses(step=10)

plt.figure()
msd, time = analysis.get()
for m in msd:
    plt.plot(time, m, color="C0", alpha=0.3)
plt.plot(time, np.mean(msd, axis=0), color="C1", marker='.')

plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mdstuff.dynamics.helpers import msd_fft
#
# N = 100000
# M = 50
# L = N // M // 4
# K = N // M
#
# ave_msd = np.zeros(L)
#
# x = np.cumsum(np.random.randn(N))
# for j in range(M):
#     msd = msd_fft(x[j * K : (j + 1) * K], length=L)
#     ave_msd += msd / M
#     plt.plot(msd, color="C0", alpha=0.3)
#
# plt.plot(ave_msd, color="C1")
# plt.plot([0, ave_msd.size], [0, ave_msd.size], color="black", ls="--")
# plt.show()
