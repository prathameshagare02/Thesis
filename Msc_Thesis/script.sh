# /usr/bin/zsh

n=25
for (( i=1; i<=n; i++ ))
do
    python sam3-v5.py --image data/New/${i}.JPG --res_dir Exp
done