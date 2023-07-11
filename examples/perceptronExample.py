import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basicNeuralNets as BNN


ds = pd.read_csv("/sonar.csv", header=None)
f = len(ds.iloc[0]) - 1
inst = BNN.PerceptronNetwork(f, 0.02)
ds.replace(to_replace=["R", "M"], value=[-1.0, 1.0], inplace=True)
ds = BNN.normalize_bipolar(ds)
tr_set = ds.sample(frac=0.8)
tr_outs = tr_set.iloc[:, f].tolist()
tr_set.drop(columns=tr_set.columns[f])
inst.train_network(tr_set, tr_outs, 10)
inst.display_network()
te_set = ds.sample(frac=0.4)
te_outs = te_set.iloc[:, f].tolist()
te_set.drop(columns=te_set.columns[f])
acc, l = inst.test_network(te_set, te_outs)
print("Accuracy:", acc)
print("Generated responses:", l)
