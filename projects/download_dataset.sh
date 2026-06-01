#!/bin/bash
curl -L -o ../data/iris-flower-dataset.zip \
    https://www.kaggle.com/api/v1/datasets/download/arshid/iris-flower-dataset
7z x ../data/iris-flower-dataset.zip -o../data/
rm ../data/iris-flower-dataset.zip
