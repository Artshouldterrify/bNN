import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basicNeuralNets as BNN


ds = pd.read_csv("../diabetes.csv", header=None)
inst = BNN.BackpropogationNetwork(1, 10, len(ds.iloc[0])-1, 2, 0.2)
ds = BNN.normalize(ds)
training_ds = ds.sample(frac=0.8)
training_outs = training_ds.iloc[:, len(ds.iloc[0]) - 1].tolist()
training_out_vector = inst.get_out_vector(training_outs, 2)
training_ds = training_ds.drop(columns=ds.columns[len(ds.iloc[0]) - 1])
inst.train_network(training_ds, 100, training_out_vector)
inst.display_network()
testing_ds = ds.sample(frac=0.3)
testing_outs = testing_ds.iloc[:, len(ds.iloc[0])-1].tolist()
testing_out_vector = inst.get_out_vector(testing_outs, 2)
testing_ds = testing_ds.drop(columns=ds.columns[len(ds.iloc[0]) - 1])
accr, l = inst.test_network(testing_ds, testing_out_vector)
print("Accuracy:", accr)
print("Generated responses:", l)
BNN.visualize_network(inst)
