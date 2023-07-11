# basicNeuralNetworks
Test python library implementing a few basic neural network models.

***
## Installation
```
pip install -i https://test.pypi.org/simple/ basicNeuralNetworks
```

## Getting started
Refer to this document for the working of this library.

Alternatively, documentation for this package can be found at: https://test.pypi.org/project/basicNeuralNetworks/

All examples and datasets used are part of the main directory.

# Basic Neural Networks
***
A library implementing 3 basic neural networks from scratch: Perceptrons, Adaptive Linear Neurons and 
Backpropogation networks.

### 1. Backpropogation Networks
Backpropogation networks are implemented via instances of the `BackpropogationNetwork` class.
Initialize an object of this class as `BackpropogationNetwork(num_hid, hid_n, inp_n, out_n, alpha)`.

`num_hid` is an int denoting the number of hidden neuron layers.

`hid_n` is an int denoting the number of neurons per hidden layer.

`inp_n` is an int denoting the number of neurons in the input layer.

`out_n` is an int denoting the number of neurons in the output layer.

`alpha` is a float in the range [0,1] denoting the training rate of the network.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basicNeuralNets as BNN


ds = pd.read_csv("../diabetes.csv", header=None)
inst = BNN.BackpropogationNetwork(1, 10, len(ds.iloc[0])-1, 2, 0.2)
```
(Importing the diabetes dataset and creating a backpropogation network with 1 hidden layer, 
10 hidden neurons and a training rate of 0.2.)

The dataset, if needed, can be normalized using the `normalize()` method.
```python
ds = BNN.normalize(ds)
```
Training the network requires an output vector, which can be obtained from the dataset's
output label column via `obj.get_output_vector(output_list, out_num)` method. Here `output_list` is
a list of output labels corresponding to a dataset, while `out_num` denotes the number of output classes.

`normalize()` and `obj.get_output_vector()` work similarly for `AdalineNetwork` and `PerceptronNetwork`.

(**Note**: we remove the output label column from the dataframe before using the dataframe to train the network.)
```python
training_ds = ds.sample(frac=0.8)
training_outs = training_ds.iloc[:, len(ds.iloc[0]) - 1].tolist()
training_out_vector = inst.get_out_vector(training_outs, 2)
training_ds = training_ds.drop(columns=ds.columns[len(ds.iloc[0]) - 1])
```
Train the network using `obj.train_network(training_ds, epochs, training_vector)`.
`training_ds` is the training dataset with the output label column removed, `epochs` is an integer
denoting the number of cycles for which the network is to be trained and `training_vector` is the output vector
obtained previously.
```python
inst.train_network(training_ds, 100, training_out_vector)
```
The state of the network can be displayed via `obj.display_network()`. This consists of 
n-1 (assuming n layers) sets of layer weights denoting weights from layer i to i+1.
```python
inst.display_network()
```
The network can be tested using the `inst.test_network(testing_ds, testing_vector)` method,
with `testing_ds` and `testing_vector` being similar to those used previously in `obj.train_network`.

The method returns two values, an accuracy score and a list of generated responses for each testing tuple.
```python
testing_ds = ds.sample(frac=0.3)
testing_outs = testing_ds.iloc[:, len(ds.iloc[0])-1].tolist()
testing_out_vector = inst.get_out_vector(testing_outs, 2)
testing_ds = testing_ds.drop(columns=ds.columns[len(ds.iloc[0]) - 1])
accr, l = inst.test_network(testing_ds, testing_out_vector)
print("Accuracy:", accr)
print("Generated responses:", l)
```
`visualize_network()` can be used to generate a visual, graphical, matplotlib-based representation of the network.

(**Note**: This method can additionally be used in exactly the same way for `AdalineNetwork` and `PerceptronNetwork` obejcts)
```python
BNN.visualize_network(inst)
```
![example-graph](https://drive.google.com/uc?export=view&id=1GdkcNlbw_FsKugGE55nyTkwfaWyJfEcP)

***

### 2. Adaline Networks

Adaline networks are implemented via instances of the `AdalineNetwork` class.
Initialize an object of this class as `AdalineNetwork(inp_n, alpha)`.

`inp_n` is an int denoting the number of neurons in the input layer.

`alpha` is a float in the range [0,1] denoting the training rate of the network.

(**Note**: This type of network works solely on binary classification problems.)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basicNeuralNets as BNN


ds = pd.read_csv("../ionosphere.csv", header=None)
f = len(ds.iloc[0]) - 1
ds.replace(to_replace=["b", "g"], value=[-1.0, 1.0], inplace=True)
inst = BNN.AdalineNetwork(f, 0.1)
```
`df.replace` is used to change the output labels from characters to bi-polar classes.

Train the network using `obj.train_network(training_ds, training_vector, epochs)`.
`training_ds` is the training dataset with the output label column removed, `epochs` is an integer
denoting the number of cycles for which the network is to be trained.

However, `training_vector` here is **not** the same as a backpropogation network, but simply a
list of output labels corresponding to each training tuple as extracted from the dataframe.
```python
tr_set = ds.sample(frac=0.8)
tr_outs = tr_set.iloc[:, f].tolist()
tr_set.drop(columns=tr_set.columns[f])
inst.train_network(tr_set, tr_outs, 100)
```
The network can be tested using the `inst.test_network(testing_ds, testing_vector)` method,
with `testing_ds` and `testing_vector` being similar to those used previously in `obj.train_network` above (and not a backpropogation network).

The method returns two values, an accuracy score and a list of generated responses for each testing tuple.
```python
te_set = ds.sample(frac=0.4)
te_outs = te_set.iloc[:, f].tolist()
te_set.drop(columns=te_set.columns[f])
acc, l = inst.test_network(te_set, te_outs)
print("Accuracy:", acc)
print("Generated responses:", l)
```

***

### 3. Perceptron Networks

Perceptron networks are implemented via instances of the `PerceptronNetwork` class.
Initialize an object of this class as `PerceptronNetwork(inp_n, alpha)`.

`inp_n` is an int denoting the number of neurons in the input layer.

`alpha` is a float in the range [0,1] denoting the training rate of the network.

(**Note**: This type of network works solely on binary classification problems and requires
features to be on a bipolar scale, hence the use of `BNN.normalize_bipolar()`)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import basicNeuralNets as BNN


ds = pd.read_csv("../sonar.csv", header=None)
f = len(ds.iloc[0]) - 1
inst = BNN.PerceptronNetwork(f, 0.02)
ds.replace(to_replace=["R", "M"], value=[-1.0, 1.0], inplace=True)
ds = BNN.normalize_bipolar(ds)
```
Train the network using `obj.train_network(training_ds, training_vector, epochs)`.
`training_ds` is the training dataset with the output label column removed, `epochs` is an integer
denoting the number of cycles for which the network is to be trained.

However, `training_vector` here is **not** the same as a backpropogation network, but simply a
list of output labels corresponding to each training tuple as extracted from the dataframe.
```python
tr_set = ds.sample(frac=0.8)
tr_outs = tr_set.iloc[:, f].tolist()
tr_set.drop(columns=tr_set.columns[f])
inst.train_network(tr_set, tr_outs, 100)
```
The network can be tested using the `inst.test_network(testing_ds, testing_vector)` method,
with `testing_ds` and `testing_vector` being similar to those used previously in `obj.train_network` above (and not a backpropogation network).

The method returns two values, an accuracy score and a list of generated responses for each testing tuple.
```python
te_set = ds.sample(frac=0.4)
te_outs = te_set.iloc[:, f].tolist()
te_set.drop(columns=te_set.columns[f])
acc, l = inst.test_network(te_set, te_outs)
print("Accuracy:", acc)
print("Generated responses:", l)
```


