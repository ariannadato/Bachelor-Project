import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#-------------------------------------------------------------------------------------
# Plot HADDOCK scores of complexes per nanobody


# This files contains HADDOCK EM scoring of a pdb_ensemble of 1 Nb (6rqm) against all targets
df = pd.read_csv('/content/drive/MyDrive/1.Thesis/emscoring_6rqm.tsv', sep='\t') # Change file name based on Nb


target_ids = ["p43366", "p21926", "p42212", "p16410", "p08"]
target_scores = {target: [] for target in target_ids}

for index, row in df.iterrows():
    for target in target_ids:
        if target in row["original_name"]:
            target_scores[target].append(row["score"])

# Boxplots

plt.figure(figsize=(6, 6))
list_names = ["p08962", "p16410", "p21926 (true)", "p42212", "p43366"]
for i, target in enumerate(target_ids):
    scores = target_scores[target]
    sns.boxplot(x=[i+1] * len(scores), y=scores)

plt.xlabel("Targets")
plt.ylabel("Haddock Score (sym-log scale)")
plt.yscale("symlog")
plt.title("Boltz Models 6 rqm + targets")
plt.xticks(range(1, len(list_names) + 1), list_names)
plt.savefig("boltz_6z1z", dpi=300, transparent=True)
plt.show()

# Variance Calculation
for target in target_ids:
    print(f"Variance for {target}: {np.var(target_scores[target])}")
