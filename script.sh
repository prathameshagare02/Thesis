#!/bin/zsh

n=62

for (( i=1; i<=n; i++ ))
do
    python sam3-exp-2-stage.py --image data/New/${i}.JPG --res_dir exp-2-stage --no_show
done