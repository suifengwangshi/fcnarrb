#!/usr/bin/bash

Datadir=${1}
ModelPath=${2}
Channels=${3}
for experiment in $(ls ./$Datadir/)
do
    echo "working on $experiment."
    if [! -d ./model/$ModelPath/$experiment ]; then
        mkdir -p ./model/$ModelPath/$experiment
    else
        continue
    fi
   
    python run_motif.py -d `pwd`/$Datadir/$experiment/data \
                         -n $experiment \
                         -g 0 \
                         -b 100 \
                         -lr 0.001 \
                         -t 0.9 \
                         -e 20 \
                         -w 0.0005 \
                         -c `pwd`/model/$ModelPath/$experiment \
                         -rl 0.3 \
                         -ch $Channels
done
