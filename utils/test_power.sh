#!/bin/bash
# Monitor power consumption for llama-bench with different thread configurations
# Usage: ./monitor_power.sh <model_path> <output_csv> <pp_threads> <tg_threads>
# Example: ./monitor_power.sh models/model.gguf results.csv "1,2,4,8" "1,2,4,8"

set -e

# Parse arguments
if [ $# -ne 4 ]; then
    echo "Usage: $0 <model_path> <output_csv> <pp_threads> <tg_threads>"
    echo "Example: $0 models/model.gguf results.csv \"1,2,4,8\" \"1,2,4,8\""
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_CSV="$2"
PP_THREADS="$3"
TG_THREADS="$4"

TEMP_LOG="/tmp/power_monitor_$$.log"
PID_FILE="/tmp/monitor_$$.pid"
BENCH_OUTPUT="/tmp/bench_output_$$.txt"

# Validate model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Function to monitor CPU stats
monitor_cpu() {
    local log_file="$1"
    echo "Timestamp,CPU_Usage(%),Avg_Freq(MHz)" > "$log_file"
    while [ -f "$PID_FILE" ]; do
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print 100-$8}')
        avg_freq=$(grep "cpu MHz" /proc/cpuinfo | awk '{sum+=$4; count++} END {printf "%.0f", sum/count}')
        timestamp=$(date +%s.%N)
        echo "$timestamp,$cpu_usage,$avg_freq" >> "$log_file"
        sleep 0.5
    done
}

# Function to calculate average power
calculate_power() {
    local log_file="$1"
    awk -F',' 'NR>1 {sum_cpu+=$2; count++} END {
        if (count > 0) {
            avg_cpu = sum_cpu/count
            est_power = avg_cpu * 200 / 100
            printf "%.2f", est_power
        } else {
            print "0"
        }
    }' "$log_file"
}

# Function to extract throughput from llama-bench output
extract_throughput() {
    local bench_output="$1"
    local workload="$2"
    grep "$workload" "$bench_output" | awk '{
        # Extract mean from "mean ± std" format
        for (i=1; i<=NF; i++) {
            if ($(i+1) == "±") {
                printf "%.2f", $i
                exit
            }
        }
    }'
}

# Function to run single benchmark
run_benchmark() {
    local workload="$1"  # "pp" or "tg"
    local threads="$2"
    local n_flag=""
    
    if [ "$workload" = "pp" ]; then
        n_flag="-n 0"
        workload_name="pp128"
    else
        n_flag="-n 128"
        workload_name="tg128"
    fi
    
    # Output progress to stderr (won't be captured in CSV)
    echo "Testing $workload_name with $threads threads..." >&2
    
    # Start monitoring
    touch "$PID_FILE"
    monitor_cpu "$TEMP_LOG" &
    local monitor_pid=$!
    
    # Run benchmark
    ./build/bin/llama-bench -m "$MODEL_PATH" -p 128 $n_flag -t "$threads" -ngl 0 > "$BENCH_OUTPUT" 2>&1
    
    # Stop monitoring
    rm -f "$PID_FILE"
    wait $monitor_pid 2>/dev/null || true
    
    # Extract results
    local throughput=$(extract_throughput "$BENCH_OUTPUT" "$workload_name")
    local power=$(calculate_power "$TEMP_LOG")
    
    if [ -z "$throughput" ] || [ "$throughput" = "0" ]; then
        echo "Warning: Failed to extract throughput for $workload_name, threads=$threads" >&2
        throughput="0"
    fi
    
    # Calculate J/t (Joules per token)
    local j_per_token=$(awk -v p="$power" -v t="$throughput" 'BEGIN {
        if (t > 0) printf "%.4f", p/t; else print "0"
    }')
    
    # Output progress to stderr
    echo "  Throughput: $throughput t/s, Power: $power W, Energy: $j_per_token J/t" >&2
    
    # Only output CSV line to stdout (this will be captured)
    echo "$workload_name,$threads,$throughput,$power,$j_per_token"
}

# Initialize CSV
echo "Workload,Threads,Throughput(t/s),Power(W),Energy(J/t)" > "$OUTPUT_CSV"

# Test PP workloads
IFS=',' read -ra PP_ARRAY <<< "$PP_THREADS"
for threads in "${PP_ARRAY[@]}"; do
    threads=$(echo "$threads" | xargs)  # trim whitespace
    result=$(run_benchmark "pp" "$threads")
    echo "$result" >> "$OUTPUT_CSV"
done

# Test TG workloads
IFS=',' read -ra TG_ARRAY <<< "$TG_THREADS"
for threads in "${TG_ARRAY[@]}"; do
    threads=$(echo "$threads" | xargs)  # trim whitespace
    result=$(run_benchmark "tg" "$threads")
    echo "$result" >> "$OUTPUT_CSV"
done

# Cleanup
rm -f "$TEMP_LOG" "$BENCH_OUTPUT" "$PID_FILE"

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""
cat "$OUTPUT_CSV"
