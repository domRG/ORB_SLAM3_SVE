#!/bin/bash

trap 'echo exit signal received!; kill $(jobs -p) &> /dev/null; wait' EXIT

read -s -p "Enter Password for sudo: " sudo_pw
echo ""

test_dir=$1
for i in $(seq 92 99)
do
	echo -e "Run: ${i}"
	for type in "${test_dir}"/*
	do
		IFS="/" read -ra type_parts <<< "${type}"
		type_len=${#type_parts[@]}
		echo -e "\tType: ${type_parts[type_len-1]}"
		
		for bag in "${type}"/*
		do
			if [ "${bag: -4:4}" == ".bag" ]; then
				IFS="/" read -ra path_parts <<< "${bag}"
				path_len=${#path_parts[@]}
				echo -e "\t\tBag: ${path_parts[path_len-1]}"
				echo "${bag:0:-4}_${i}" >> slam_log.txt

				current_window=$(xdotool getactivewindow)

				rosrun ORB_SLAM3 Mono_Inertial ~/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt ~/Documents/fyp/datasets/euroc/euroc_monoi.yaml >>slam_log.txt 2>&1 &
				slam_pid=$!
				echo $sudo_pw | sudo -S renice -n -20 $slam_pid &> /dev/null

				WINDOWS=()
				while [ ${#WINDOWS[@]} -lt 2 ]; do
					# get WINDOWS array of orbslam3 windows
					eval $(xdotool search --onlyvisible --shell --name "ORB-SLAM3: ")
					sleep 0.1
				done

				for id in ${WINDOWS[@]}
				do
					{ # try
						xdotool windowminimize ${id}
					}||{ # catch
						# do nothing
						a=1
					}
				done
				xdotool windowactivate ${current_window}

				sleep 2
				rosbag play $bag &> /dev/null &
				bag_pid=$!

				wait $bag_pid
				sleep 1
				
				kill -INT $slam_pid >>slam_log.txt 2>&1
				
				wait $slam_pid >>slam_log.txt 2>&1
				sleep 1
				
				mv ros_output_traj.txt "${bag:0:-4}_traj_${i}.txt" >>slam_log.txt 2>&1
				mv ros_output_sve.txt "${bag:0:-4}_sve_${i}.txt" >>slam_log.txt 2>&1
			fi
		done
	done
done
