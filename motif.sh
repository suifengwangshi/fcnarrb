#!/usr/bin/bash

Data=${1}
method=${2}
model=${3}
for experiment in $(ls ./${Data}/)
do
    echo "working on ${experiment}."
    if [ ! -d ./motifs/$method/${experiment} ]; then
        mkdir -p ./motifs/$method/${experiment}
    else
        continue
    fi
    
    python motif_finder.py -d `pwd`/${Data}/${experiment}/data \
                           -n ${experiment} \
                           -t 0.9 \
                           -g 0 \
                           -c `pwd`/model/$model/${experiment} \
                           -o `pwd`/motifs/$method/${experiment}
done
