from pathlib import Path
import sys

from genetic_optimizer import Gen_Optimizer

current_file = Path(__file__).resolve()
gmo_final_path = current_file.parent.parent
sys.path.append(str(gmo_final_path))



'''Note: Memoization will not work unless the user's account is specified.
This is because the memos are saved in "Documents." I did this to ensure that my
OneDrive didn't explode with memo size'''
genc = Gen_Optimizer(sys_username="Eric Arnold",
                     mode = "clustomers",
                     uniform=True, pop_cap=100, toprint=True,
                     mut_rate=0.005, metrics=["al_sep", "sil"])
genc.genetically_cluster(n_workers=12, cycles=1)
genc.write_clusters_parquet(to_max = "CH-I")