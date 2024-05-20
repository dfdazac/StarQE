#!/bin/bash

# Input arguments
dataset=$1
model_file=$2
metrics=$3
query_types=$4

special_queries=("doublepath" "triangle" "square")

# Function to extract the value for a given metric from the evaluation result
extract_metric_value() {
    local metric=$1
    local result=$2
    echo "$result" | grep -Po "(?<='$metric': )[0-9.]+" | head -1
}


IFS=',' read -ra queries <<< "$query_types"
# Prepare to store results
declare -A results

# Run evaluation for each query type and store the results
for q in "${queries[@]}"; do
    echo "Evaluating on $q queries" >&2

	if [[ " ${special_queries[@]} " =~ " $q " ]]; then
        test_data="/$q/*:*"
    else
        test_data="/$q/unanchored:*"
    fi

    result=$(hqe evaluate --dataset=$dataset --test-data="$test_data" --num-workers=2 --batch-size=8 --model-path=$(pwd)/models/$model_file 2>&1 | tee /dev/tty)
    for metric in $(echo $metrics | tr "," "\n"); do
        value=$(extract_metric_value $metric "$result")
        results["$metric,$q"]=$(printf "%.3f" $value)
    done
done

# Prepare header for the table
echo '============== RESULTS =============='
echo -n " & "
for q in "${queries[@]}"; do
    echo -n "$q & "
done
echo

# Print metrics and results in table format
IFS=',' read -ra metrics_arr <<< "$metrics"
for metric in "${metrics_arr[@]}"; do
    echo -n "$metric & "
    for q in "${queries[@]}"; do
        echo -n "${results["$metric,$q"]} & "
    done
    echo
done

