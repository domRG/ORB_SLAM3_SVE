#!/bin/bash

trap 'echo exit signal received!; kill $(jobs -p) &> /dev/null; wait' EXIT

read -s -p "Enter Password for sudo: " sudo_pw
echo ""

test_dir=$1
for i in $(seq 0 24)
do
	n=$((i*4))	
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

				rosrun ORB_SLAM3 Mono_Inertial SlamNode0 ~/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt ~/Documents/fyp/datasets/euroc/euroc_monoi.yaml >>slam_log.txt 2>&1 &
				slam0_pid=$!
				echo $sudo_pw | sudo -S renice -n -20 $slam0_pid &> /dev/null

				rosrun ORB_SLAM3 Mono_Inertial SlamNode1 ~/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt ~/Documents/fyp/datasets/euroc/euroc_monoi.yaml >>slam_log.txt 2>&1 &
				slam1_pid=$!
				echo $sudo_pw | sudo -S renice -n -20 $slam1_pid &> /dev/null

				rosrun ORB_SLAM3 Mono_Inertial SlamNode2 ~/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt ~/Documents/fyp/datasets/euroc/euroc_monoi.yaml >>slam_log.txt 2>&1 &
				slam2_pid=$!
				echo $sudo_pw | sudo -S renice -n -20 $slam2_pid &> /dev/null

				rosrun ORB_SLAM3 Mono_Inertial SlamNode3 ~/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt ~/Documents/fyp/datasets/euroc/euroc_monoi.yaml >>slam_log.txt 2>&1 &
				slam3_pid=$!
				echo $sudo_pw | sudo -S renice -n -20 $slam3_pid &> /dev/null

				WINDOWS=()
				while [ ${#WINDOWS[@]} -lt 8 ]; do
					# get WINDOWS array of orbslam3 windows
					eval $(xdotool search --onlyvisible --shell --name "ORB-SLAM3: " 2> /dev/null)
					sleep 0.1
				done

				xdotool windowactivate ${current_window}
				# do twice just to catch missed ones :/
				for j in 1 2; do
					for id in ${WINDOWS[@]}
					do
						{ # try
							xdotool windowminimize ${id} &> /dev/null
						}||{ # catch
							# do nothing
							a=1
						}
					done
				done

				sleep 2
				rosbag play $bag &> /dev/null &
				bag_pid=$!

				wait $bag_pid
				sleep 1
				
				kill -INT $slam0_pid >>slam_log.txt 2>&1
				
				wait $slam0_pid >>slam_log.txt 2>&1
				sleep 1
				
				mv ros_output_traj.txt "${bag:0:-4}_traj_$((n+0)).txt" >>slam_log.txt 2>&1
				mv ros_output_sve.txt "${bag:0:-4}_sve_$((n+0)).txt" >>slam_log.txt 2>&1


				kill -INT $slam1_pid >>slam_log.txt 2>&1
				
				wait $slam1_pid >>slam_log.txt 2>&1
				sleep 1
				
				mv ros_output_traj.txt "${bag:0:-4}_traj_$((n+1)).txt" >>slam_log.txt 2>&1
				mv ros_output_sve.txt "${bag:0:-4}_sve_$((n+1)).txt" >>slam_log.txt 2>&1


				kill -INT $slam2_pid >>slam_log.txt 2>&1
				
				wait $slam2_pid >>slam_log.txt 2>&1
				sleep 1
				
				mv ros_output_traj.txt "${bag:0:-4}_traj_$((n+2)).txt" >>slam_log.txt 2>&1
				mv ros_output_sve.txt "${bag:0:-4}_sve_$((n+2)).txt" >>slam_log.txt 2>&1


				kill -INT $slam3_pid >>slam_log.txt 2>&1
				
				wait $slam3_pid >>slam_log.txt 2>&1
				sleep 1
				
				mv ros_output_traj.txt "${bag:0:-4}_traj_$((n+3)).txt" >>slam_log.txt 2>&1
				mv ros_output_sve.txt "${bag:0:-4}_sve_$((n+3)).txt" >>slam_log.txt 2>&1
			fi
		done
	done
done
