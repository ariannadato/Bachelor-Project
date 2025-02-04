import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

# Extract relevant data from AlphaFold3 output folders

filepath = Path("/content/drive/MyDrive/1.Thesis/6_AF3_cluster/true_pairs")

summ_data = {}
for nb_target in filepath.iterdir():
  if nb_target.is_dir():
    nb_target_name = str(nb_target).split('/')[-1]
    print(nb_target_name)
    for seed_folder in nb_target.iterdir():
      if seed_folder.is_dir():
        seed_folder_name = str(seed_folder).split('/')[-1]
        for summ_file in seed_folder.iterdir():
          if summ_file.name == "summary_confidences.json":
            summ_file_name = f"{nb_target_name}_{seed_folder_name}"
            summ_file_content = pd.read_json(summ_file)
            summ_data[summ_file_name] = summ_file_content


# Plot ipTM

plot_iptm = []
for key, df in summ_data.items():
    if key.split('_')[0] == "emc":
      nb_target_name = key.split('_')[0] + "_" + key.split('_')[1] + "_" + key.split('_')[2]
    else:
      nb_target_name = key.split('_')[0] + "_" + key.split('_')[1]
    for value in df['iptm']:
        plot_iptm.append({'nb_target': nb_target_name, 'iPTM': value})

print(plot_iptm)
plot_df = pd.DataFrame(plot_iptm)

plt.figure(figsize=(10, 6))
plot_df.boxplot(by='nb_target', column='iPTM', grid=False)
plt.title("iPTM for true nb_target pairs")
plt.suptitle('')  # Remove the default subtitle
plt.xlabel('nb_target')
plt.ylabel('iPTM')
plt.xticks(rotation= 60)
plt.tight_layout()
plt.savefig("IPTM_True.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# Plot pTM

plot_ptm = []
for key, df in summ_data.items():
    if key.split('_')[0] == "emc":
      nb_target_name = key.split('_')[0] + "_" + key.split('_')[1] + "_" + key.split('_')[2]
    else:
      nb_target_name = key.split('_')[0] + "_" + key.split('_')[1]
    for value in df['ptm']:
        plot_ptm.append({'nb_target': nb_target_name, 'PTM': value})

plot_df2 = pd.DataFrame(plot_ptm)

plt.figure(figsize=(10, 6))
plot_df2.boxplot(by='nb_target', column='PTM', grid=False)
plt.title('PTM for true nb_target complexes')
plt.suptitle('')  # Remove the default subtitle
plt.xlabel('nb_target')
plt.ylabel('PTM')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Plot pTM and ipTM in the same graph for visual comparison

categories = plot_df['nb_target'].unique()
categories.sort()


iptm_means = [] # Compute statistics for iPTM
iptm_spreads = []

for category in categories:
    data = plot_df[plot_df['nb_target'] == category]['iPTM']
    iptm_means.append(data.mean())
    iptm_spreads.append(data.std())  

ptm_means = []   # Compute statistics for PTM
ptm_spreads = []

for category in categories:
    data = plot_df2[plot_df2['nb_target'] == category]['PTM']
    ptm_means.append(data.mean())
    ptm_spreads.append(data.std())  

# Create bar chart
x = np.arange(len(categories))  
width = 0.4  # Bar width

plt.figure(figsize=(12, 6))

# Plot bars
plt.bar(x - width / 2, iptm_means, width, yerr=iptm_spreads, label='iPTM', color='lightblue', capsize=5)
plt.bar(x + width / 2, ptm_means, width, yerr=ptm_spreads, label='PTM', color='lightgreen', capsize=5)
plt.xticks(x, categories, rotation=60)
plt.xlabel('nb_target')
plt.ylabel('Values')
plt.title("PTM / iPTM of true complexes")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("PTM_iPTM.png", dpi = 300, transparent = True)
plt.show()
