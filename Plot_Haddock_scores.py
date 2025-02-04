import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Extract relevant data

dataframes = {}
for filen in os.listdir("/content/drive/MyDrive/1.Thesis/6_AF3_cluster/true_pairs/7_haddock_af3"):
  if filen.endswith(".tsv"):
    name = os.path.splitext(filen)[0]
    name = name.replace("emscoring_", "")
    dataframes[name] = pd.read_csv("/content/drive/MyDrive/1.Thesis/6_AF3_cluster/true_pairs/7_haddock_af3/"+filen, sep="\t")
    print(name)

# Plot in a boxplot

for name, df in dataframes.items():
    df.columns = df.columns.str.strip()  

boxplot_data = [df['score'] for df in dataframes.values()]
plt.figure(figsize=(12, 6))
plt.boxplot(boxplot_data, labels=dataframes.keys(), vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
plt.xlabel("emscoring_nanobody_target")
plt.ylabel("Haddock Score")
plt.title("Haddock EM scores of true Nb / Target complexes")
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("boxplot_true_haddock")
plt.show()
