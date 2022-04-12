#!/usr/bin/bash
echo "cleaning data"
python scripts/clean.py -i data/raw.csv -o data/clean.csv

echo "split into train & test"
python scripts/split.py -i data/clean.csv -o data -p params.yaml

echo "train model"
python scripts/train.py -i data/train.csv -o models

echo "evaluate fit"
python scripts/eval.py --data data --models models --metrics metrics.json

echo "create plot"
python scripts/plots.py --data data --models models --plots plots