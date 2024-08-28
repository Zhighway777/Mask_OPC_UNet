#!/bin/bash

total_l2=0
total_pvb=0
count=0
Runtime=129

for i in {1..10}
do
	output=$(python ./LithoSim/eval.py --mask_file_name="M1_test${i}_res.png" --layout_file_name="M1_test${i}.png" | grep 'L2 =')

echo "$output"
# 使用awk从输出中提取L2和PVB的值，并累加
    l2_value=$(echo $output | awk '{print $4}')  # 假设L2值是第4个字段
    pvb_value=$(echo $output | awk '{print $8}')  # 假设PVB值是第6个字段
    total_l2=$((total_l2 + l2_value))
    total_pvb=$((total_pvb + pvb_value))
    ((count++))

done

average_l2=$((total_l2 / count))
average_pvb=$((total_pvb / count))
echo "Average L2: $average_l2"
echo "Average PVB: $average_pvb"
echo "Total_Runtime: $Runtime"
cost=$((100 * Runtime + total_l2 + total_pvb))
echo "Cost = Sum{For all cases} 100*Runtime+L2+PVBand :"
echo "$cost"
