#!/bin/bash



for i in {1..10}
do
	output=$(python ./LithoSim/eval.py --mask_file_name="M1_test${i}_res.png" --layout_file_name="M1_test${i}.png" | grep 'L2 =')
	echo "$output"
	l2_value=$(echo $output | awk '{print $4}')
	pvb_value=$(echo $output | awk '{print $8}')
	echo "$l2_value"
	echo "$pvb_value"
done	


