#!/bin/bash
input_dir=$1
output_dir=$2
test_dir=$3
cluster=$4
minHess=$5
maxCount=$6
attempts=$7
C=$8
gamma=$9

echo $maxCount
echo $attempts
echo $C
echo $gamma


#################################################################################
#################Tworzenie drzewa################################################
#################################################################################

vocab=${output_dir}/vocab_${cluster}_${minHess}_${maxCount}_${attempts}.xml
svm=${output_dir}/svm_${cluster}_${minHess}_${maxCount}_${attempts}_${C}_${gamma}.xml
log=${output_dir}/log_${cluster}_${minHess}_${maxCount}_${attempts}_${C}_${gamma}.txt
echo $vocab
echo $svm
echo $log



./default_trainer ${input_dir} $vocab ${cluster} ${minHess} 2 2 1 1 ${maxCount} ${attempts} $log
./default_svm ${input_dir} $vocab $svm ${cluster} ${minHess} 2 2 1 1 ${maxCount} ${attempts} ${C} ${gamma} $log
./default_predict ${test_dir} $vocab $svm ${cluster} ${minHess} 2 2 1 1 ${maxCount} ${attempts} $log
