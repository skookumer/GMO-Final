from pathlib import Path
import sys

from genetic_optimizer import Gen_Optimizer

current_file = Path(__file__).resolve()
gmo_final_path = current_file.parent.parent
sys.path.append(str(gmo_final_path))

'''Note: Memoization will not work unless the user's account is specified.
This is because the memos are saved in "Documents." I did this to ensure that my
OneDrive didn't explode with memo size'''

geno = Gen_Optimizer(sys_username="Eric Arnold",
                    smoothing_alpha=0, uniform=True, mode="rulemine", 
                     pop_cap=1000, mut_rate=0.1, k_range=(2, 3), 
                     toprint=True, metrics=["sup", "conf"])
geno.genetically_modify(cycles=1)
geno.dot_plot_popn()
geno.iiid_plot_popn()
geno.iiid_plot_multi(["minsup 20, mut 0.1, 30 trials, sup conf, VL/memo_1.jsonl", "memo_1.jsonl"])
geno.write_rules_csv()