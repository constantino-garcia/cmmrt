import numpy as np
import pandas as pd

from cmmrt.rt.data import load_cmm_fingerprints, PredRetFeaturizedSystem, AlvadescDataset
from cmmrt.projection.data import load_predret


# Systems with larger number of available fingerprints are
# 'Waters ACQUITY UPLC with Synapt G1 Q-TOF'  or 'RIKEN' (as commented code proves). Let's use RIKEN to avoid introducing
# a new system in the paper
# cmm_fingerprints = load_cmm_fingerprints()
# cmm_fingerprints = cmm_fingerprints[cmm_fingerprints.pid != "\\N"].astype({'pid': 'int'})
# cmm_pids = cmm_fingerprints.pid.values
# counts = []
# for system in np.unique(predret.System.values):
#     system_data = predret.loc[predret["System"] == system]
#     merged = np.intersect1d(system_data.Pubchem.values.astype('int'), cmm_pids)
#     counts.append({'system': system, 'count': len(merged)})
# pd.DataFrame(counts).sort_values('count')

riken = PredRetFeaturizedSystem('RIKEN')
