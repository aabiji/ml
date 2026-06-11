import struct
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

def load_iris():
    df = pd.read_csv("../data/IRIS.csv")
    df = pd.get_dummies(df, columns=["species"], dtype=int)
    df = df.sample(frac = 1)

    in_columns = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    in_values = torch.tensor(in_columns.values, dtype=torch.float32)
    x_train, x_test = in_values[0:100], in_values[100:]

    out_columns = df[["species_Iris-setosa", "species_Iris-versicolor", "species_Iris-virginica"]]
    out_values = torch.tensor(out_columns.values, dtype=torch.float32)
    y_train, y_test = out_values[0:100], out_values[100:]
    return x_train, y_train, x_test, y_test

def load_mnist(train_input_path, train_label_path, test_input_path, test_label_path):
    inputs, labels = [], []

    for path in [train_input_path, test_input_path]:
        with open(path, "rb") as file:
            _, size = struct.unpack(">II", file.read(8))
            height, width = struct.unpack(">II", file.read(8))
            buf = np.frombuffer(file.read(), dtype=np.dtype(np.uint8).newbyteorder(">"))
            buf = np.reshape(buf, (size, height * width))
            tensor = torch.tensor(buf, dtype=torch.float32)
            inputs.append(tensor / 255.0)

    for path in [train_label_path, test_label_path]:
        with open(path, "rb") as file:
            _, size = struct.unpack(">II", file.read(8))
            buf = np.frombuffer(file.read(), dtype=np.dtype(np.uint8).newbyteorder(">"))
            tensor = torch.tensor(buf, dtype=torch.long)
            one_hot = F.one_hot(tensor, num_classes=10)
            labels.append(one_hot.float())

    return inputs[0], labels[0], inputs[1], labels[1]

def to_nparray(x_train, y_train, x_test, y_test):
    return x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()