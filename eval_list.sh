#!/bin/bash

list=("" "_alt001" "_alt002" "_alt002_v2" "_alt003" "_alt003_v2" "_alt003_v3" "_alt004" "_alt004_v2" "_alt005" "_alt005_vdiv2" "_alt006")

for n in ${list[@]}
do
	python3 ./eval/eval.py ~/Documents/fyp/datasets/euroc/MH04/mav0/state_groundtruth_estimate0/data.csv ./mh04"$n"_ros_output_traj.txt --verbose mh04"$n"_verb.txt --plot ./mh04"$n"_ros_traj_plot --sve ./mh04"$n"_ros_output_sve.txt ./mh04"$n"_sve_plot
done
