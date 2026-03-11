#!/bin/zsh

n=62

for (( i=1; i<=n; i++ ))
do
    # python sam3-exp-2-stage.py --image data/New/${i}.JPG --res_dir exp-2-stage-wo-cal --no_show
    # Try even simpler prompts
    # python sam3-simple.py --image data/New/${i}.JPG --res_dir MORE-RUST-PROMPTS --no_show
    # python sam3-final.py --image data/New/${i}.JPG --res_dir NON-CAL-SIMPLE-SAM --no_show
    # python from\ __future__\ import\ annotations.py --image data/New/${i}.JPG --show 0
    python ResNet_FE.py --image data/New/${i}.JPG --show 0 
done