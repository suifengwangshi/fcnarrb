#!/usr/bin/bash

data_path=${1}
for experiment in $(ls ${data_path}/)
do
  {
    echo ${experiment}
    testPath=`pwd`/${data_path}/${experiment}/data
    if [[ -d ${testPath} ]]; then
    {
      rm -rf ${testPath}
    }
    fi
  }
done

