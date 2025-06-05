#!/bin/bash
cd ..
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --root)
      ROOT="$2"
      shift; shift
      ;;
    --backbone)
      BACKBONE="$2"
      shift; shift
      ;;
    --n_shots)
      N_SHOTS="$2"
      shift; shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Check that all required arguments are provided
if [[ -z "$ROOT" || -z "$BACKBONE" || -z "$N_SHOTS" ]]; then
  echo "Usage: bash run_datasets.sh --root /path/to/data --backbone BACKBONE_NAME --n_shots N"
  exit 1
fi

# List of datasets (excluding 'imagenet')
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

# Loop over each dataset
for dataset in "${datasets[@]}"; do
  echo "Running: $dataset | Backbone: $BACKBONE | Shots: $N_SHOTS"
  python main.py --dataset "$dataset" --root_path "$ROOT" --backbone "$BACKBONE" --n_shots "$N_SHOTS" --seed 1 --source_prompts_types imagenet_images
done


