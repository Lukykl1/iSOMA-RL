import pandas as pd
import os

#Dictionary to store statistics data
statistics = {}

#Get all files in directory with specific ending
path = './log_various_history23_nosde'
files = [f for f in os.listdir(path) if f.endswith('results.txt')]
print(path)
grouped_data = None
for file in files:
    #Read data from file
    data = pd.read_csv(os.path.join(path, file), sep="\t", header=None, names=["Function", "Result"])

    #set all numbers less than 10**-8 to 0
    data['Result'] = data['Result'].apply(lambda a:float(a)).apply(lambda x: 0 if x < 10**-8 else x)
    #Group by function name
    grouped_data = data.groupby("Function")

    #Calculate statistics data
    mean = grouped_data["Result"].mean()
    median = grouped_data["Result"].median()
    min = grouped_data["Result"].min()
    max = grouped_data["Result"].max()
    std = grouped_data["Result"].std()

    #Store statistics data in dictionary
    statistics[file] = {"Mean": mean, "Count": grouped_data["Result"].count(), "Count of functions": len(grouped_data)}

#Compare statistics between files
for file in statistics:
    print("Statistics for file: " + file)
    print(statistics[file])

import pandas as pd
import os

#Dictionary to store statistics data
statistics2 = {}

#Get all files in directory with specific ending
path = './log'
files = [f for f in os.listdir(path) if f.endswith('results.txt')]


grouped_data2 = None
print("log")
for file in files:
    #Read data from file
    data = pd.read_csv(os.path.join(path, file), sep="\t", header=None, names=["Function", "Result"])

    #set all numbers less than 10**-8 to 0
    data['Result'] = data['Result'].apply(lambda x: 0 if x < 10**-8 else x)

    #Group by function name
    grouped_data2 = data.groupby("Function")

    #Calculate statistics data
    mean = grouped_data2["Result"].mean()
    median = grouped_data2["Result"].median()
    min = grouped_data2["Result"].min()
    max = grouped_data2["Result"].max()
    std = grouped_data2["Result"].std()

    #Store statistics data in dictionary
    statistics2[file] = {"Mean": mean,  "Count": grouped_data2["Result"].count(), "Count of functions": len(grouped_data2)}

#Compare statistics between files
for file in statistics2:
    print("Statistics for file: " + file)
    print(statistics2[file])

from scipy.stats import ranksums


common_index = statistics2["isoma_results.txt"]['Mean'].index.intersection(statistics["results.txt"]['Mean'].index)
df1 = statistics2["isoma_results.txt"]['Mean'].loc[common_index]
df2 = statistics["results.txt"]['Mean'].loc[common_index]

worse =0 
better = 0
for name, group1 in grouped_data:
    if name not in grouped_data2.groups:
        continue
    group2 = grouped_data2.get_group(name)
    _, p_value = ranksums(group1['Result'], group2['Result'])
    if(p_value < 0.05 and (group1.mean()< group2.mean()).bool()):
        better += 1
    if(p_value < 0.05 and (group1.mean() > group2.mean()).bool()):
        worse += 1
    print(f'Function: {name}, p-value: {p_value}, Better: {(group1.mean()< group2.mean()).bool()} Worse: {(group1.mean() > group2.mean()).bool()}')

print(f'Worse: {worse}, Better: {better}')

print(statistics["results.txt"]['Count'][0:30])
print(statistics["results.txt"]['Count'][30:])
print("+", (df1 > df2).sum().sum(), "-", (df1 < df2).sum().sum(), "=", (df1 == df2).sum().sum())
print(df1.round(decimals = -1) >= df2.round(decimals = -1))
print( "+" ,(df1.round(decimals = -1) > df2.round(decimals = -1)).sum().sum(), "-" ,(df1.round(decimals = -1) < df2.round(decimals = -1)).sum().sum(), "=" ,(df1.round(decimals = -1) == df2.round(decimals = -1)).sum().sum())

