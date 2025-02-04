import pandas as pd
import glob
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#-----------------------------------------------------------------------------------------------------
#Extract data from Prodigy output files (1 file per model) and organize it in a dataframe

path_to_files = "/content/drive/MyDrive/1.Thesis/8_prodigy/prodigy_results/*.txt"
data = []

# Function to extract data with error handling
def extract_value(pattern, content, default_value=np.nan, cast_type=None):
    match = re.search(pattern, content)
    if match:
        value = match.group(1)
        if cast_type:
            return cast_type(value)
        return value
    else:
        return default_value


for filen in glob.glob(path_to_files):
    with open(filen, 'r') as f:
        content = f.read()

        # Extract values using the function
        intermolecular_contacts = extract_value(r'No\. of intermolecular contacts: (\d+)', content, cast_type=int)
        charged_charged_contacts = extract_value(r'No\. of charged-charged contacts: (\d+)', content, cast_type=int)
        charged_polar_contacts = extract_value(r'No\. of charged-polar contacts: (\d+)', content, cast_type=int)
        charged_apolar_contacts = extract_value(r'No\. of charged-apolar contacts: (\d+)', content, cast_type=int)
        polar_polar_contacts = extract_value(r'No\. of polar-polar contacts: (\d+)', content, cast_type=int)
        apolar_polar_contacts = extract_value(r'No\. of apolar-polar contacts: (\d+)', content, cast_type=int)
        apolar_apolar_contacts = extract_value(r'No\. of apolar-apolar contacts: (\d+)', content, cast_type=int)
        apolar_nis_percentage = extract_value(r'Percentage of apolar NIS residues: ([\d.]+)', content, cast_type=float)
        charged_nis_percentage = extract_value(r'Percentage of charged NIS residues: ([\d.]+)', content, cast_type=float)
        binding_affinity = extract_value(r'Predicted binding affinity \(kcal.mol-1\):\s+([-.\d]+)', content, cast_type=float)
        dissociation_constant = extract_value(r'Predicted dissociation constant \(M\) at 25\.0˚C:\s+([+-]?\d*\.\d+([eE][+-]?\d+)?)',content, cast_type=float)

        #only useful for display of Dataframe but not for calculation; show dissociation constant in scientific notation
    #    if isinstance(dissociation_constant, float):
    #        dissociation_constant = f"{dissociation_constant:.3e}"  
      
        # Append the data as a dictionary
        data.append({
            "File": os.path.basename(filen),
            "Intermolecular Contacts": intermolecular_contacts,
            "Charged-Charged Contacts": charged_charged_contacts,
            "Charged-Polar Contacts": charged_polar_contacts,
            "Charged-Apolar Contacts": charged_apolar_contacts,
            "Polar-Polar Contacts": polar_polar_contacts,
            "Apolar-Polar Contacts": apolar_polar_contacts,
            "Apolar-Apolar Contacts": apolar_apolar_contacts,
            "Apolar NIS Percentage": apolar_nis_percentage,
            "Charged NIS Percentage": charged_nis_percentage,
            "Binding Affinity (kcal/mol)": binding_affinity,
            "Dissociation Constant (M)": dissociation_constant
        })

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)
print(df.head())

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sort Data based on Nanobody
nanobodies = ["6z1z",
"manb9",
"emc_a7",
"emc_h6",
"7v0v",
"6rqm",
"emc_c5",
"7om5",
"6r7t",
"4qo1",
"2x1o",
"4zg1",
"5c2u",
"6f0d"]

nano_subframes = {}
for nano in nanobodies:
    nano_subframes[nano] = df[df['File'].str.contains(nano)]

#----------------------------------------------------------------------------------------------------------------------------
# Plot Prodigy metrics in boxplots

targets = ["p42212", "p00533", "p21926", "p06396", "p43366", "q9ugl1", "q7z3b4",
           "p16410", "p21860", "p04637", "q04609", "p08962"]

for nano in nano_subframes:
  for target in targets:
    if not nano_subframes[nano].empty:
      nano_subframes[nano].loc[nano_subframes[nano]['File'].str.contains(target, case=False, na=False), 'Target Name'] = target

  if 'Target Name' in nano_subframes[nano].columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=nano_subframes[nano], x='Target Name', y='Dissociation Constant (M)')
    plt.title(nano)
    plt.gca().set_yticks(plt.gca().get_yticks()[::10])

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=nano_subframes[nano], x='Target Name', y='Binding Affinity (kcal/mol)', color = "pink")
    plt.title(nano)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=nano_subframes[nano], x='Target Name', y='Intermolecular Contacts', color = "green")
    plt.title(nano)

#----------------------------------------------------------------------------------------------------------------------------------
# Calculate Binding Affinity Averages per complex (30 models per complex)

complex_averages = {}

for nano in nanobodies:
    complex_averages[nano] = {}
    for target in targets:
      if df['File'].str.contains(nano + '_' + target, case=False, na=False).any():
        complex_averages[nano][target] = df.loc[df['File'].str.contains(nano + '_' + target, case=False, na=False), "Binding Affinity (kcal/mol)"].mean()

# Display averages in scatter plot

plt.figure(figsize=(10, 6))
for nano, targets in complex_averages.items():
  print(nano, targets)
  targets_list = list(targets.keys())
  affinities_list = list(targets.values()) 
  plt.scatter(targets_list, affinities_list, label=nano)   # Scatter plot with different colors for each nanobody

plt.xlabel('Target')
plt.ylabel('Binding Affinity (kcal/mol)')
plt.title('Binding Affinity of Nanobodies against Targets')
plt.legend(title='Nanobodies', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()
