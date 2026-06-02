#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 [iris|mnist]"
    exit 1
fi


if [[ "$1" == "iris" ]]; then
    curl -L -o ../data/iris-flower-dataset.zip \
        https://www.kaggle.com/api/v1/datasets/download/arshid/iris-flower-dataset
    7z x ../data/iris-flower-dataset.zip -o../data/
    rm ../data/iris-flower-dataset.zip
fi

if [[ "$1" == "mnist" ]]; then
    curl -L -o ../data/mnist-dataset.zip \
      https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
    mkdir -p ../data/mnist
    7z x ../data/mnist-dataset.zip -o../data/mnist
    rm ../data/mnist-dataset.zip
fi
