#!/bin/bash
curr_datetime=`date +%Y%m%d%H%M%S`
job_name=predict_mnist
log_file_name=`echo ../logs/${job_name}_${curr_datetime}.log`
echo "Starting training at "${curr_datetime} >>${log_file_name}
python ../src/predict_mnist.py
if [ $? -ne 0 ]
then
    curr_datetime=`date +%Y%m%d%H%M%S`
    echo "ERROR !!! ... Training Failed at "${curr_datetime} >>${log_file_name}
    exit 1
else
    curr_datetime=`date +%Y%m%d%H%M%S`
    echo "Completed training successfully at "${curr_datetime} >>${log_file_name}
fi