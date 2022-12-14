#!/usr/bin/bash

python_path=${1}
data_path=${2}
threadnum=2
tmp="/tmp/$$.fifo"
mkfifo ${tmp}
exec 6<> ${tmp}
rm ${tmp}
for((i=0; i<${threadnum}; i++))
do
    echo ""
done >&6
for experiment in $(ls ${data_path}/)
do
  read -u6
  {
    echo ${experiment}
    python ${python_path} -d `pwd`/${data_path}/${experiment} \
                       -n ${experiment}
    
    echo "" >&6
  }&
done
wait
exec 6>&-
