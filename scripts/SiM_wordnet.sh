#!/bin/bash

# Define the root path
ROOT="/path/to/your/dataset/root"

# List of datasets
datasets=(
  "sun397"
  "fgvc_aircraft"
  "eurosat"
  "stanford_cars"
  "food101"
  "oxford_pets"
  "oxford_flowers"
  "caltech101"
  "dtd"
  "ucf101"
)

# Loop over each dataset and run the command
for dataset in "${datasets[@]}"; do
  echo "Running on dataset: $dataset"

  #if you have cached the features and target, you can add --load in the following command
  python main.py --dataset "$dataset" --root_path "$ROOT" --seed 1 --backbone vit_b16 --source_prompts_types wordnet
done
