#!/bin/bash

names=("iris" "mnist" "fane" "underwater")
kaggle_urls=(
  "arshid/iris-flower-dataset"
  "hojjatk/mnist-dataset"
  "furcifer/fane-facial-expressions-and-emotion-dataset"
  "slavkoprytula/aquarium-data-cots"
)

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <dataset-name>"
  exit 1
fi

for i in "${!names[@]}"; do
  if [[ "$1" == "${names[$i]}" ]]; then
    mkdir -p "data/${names[$i]}"
    curl -L -o "data/${names[$i]}/${names[$i]}.zip" "https://www.kaggle.com/api/v1/datasets/download/${kaggle_urls[$i]}"
    7z x "data/${names[$i]}/${names[$i]}.zip" -o"data/${names[$i]}"
    rm "data/${names[$i]}/${names[$i]}.zip"
    break
  fi
done
