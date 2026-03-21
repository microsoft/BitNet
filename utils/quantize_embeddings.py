#!/usr/bin/env python3
"""
Embedding Quantization Script
This script converts ggml-model-f32.gguf to multiple quantized versions
with different token embedding types.
"""

import subprocess
import os
import argparse
import re
import csv
from pathlib import Path
from datetime import datetime


class EmbeddingQuantizer:
    def __init__(self, input_model, output_dir, quantize_bin="../build/bin/llama-quantize", 
                 bench_bin="../build/bin/llama-bench", stats_dir="../stats", csv_output=None):
        self.input_model = Path(input_model)
        self.output_dir = Path(output_dir)
        self.quantize_bin = Path(quantize_bin)
        self.bench_bin = Path(bench_bin)
        self.stats_dir = Path(stats_dir)
        self.csv_output = Path(csv_output) if csv_output else None
        
        # Verify input file exists
        if not self.input_model.exists():
            raise FileNotFoundError(f"Input model not found: {self.input_model}")
        
        # Verify quantize tool exists
        if not self.quantize_bin.exists():
            raise FileNotFoundError(f"Quantize binary not found: {self.quantize_bin}")
        
        # Verify bench tool exists
        if not self.bench_bin.exists():
            raise FileNotFoundError(f"Benchmark binary not found: {self.bench_bin}")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.newly_created_files = set()  # Track newly created files
        
    def quantize(self, embedding_type, output_suffix):
        """
        Perform single quantization
        
        Args:
            embedding_type: Token embedding type (uppercase format, e.g., Q6_K)
            output_suffix: Output file suffix (lowercase format, e.g., q6_k)
        
        Returns:
            bool: Whether successful
        """
        output_file = self.output_dir / f"ggml-model-i2_s-embed-{output_suffix}.gguf"
        
        # Check if file already exists
        file_already_existed = output_file.exists()
        
        if file_already_existed:
            print(f"ℹ️  File already exists: {output_file}")
            print(f"   Skipping quantization, will use existing file for benchmark")
            return True
        
        cmd = [
            str(self.quantize_bin),
            "--token-embedding-type", embedding_type,
            str(self.input_model),
            str(output_file),
            "I2_S",
            "1",
            "1"
        ]
        
        print(f"\n{'='*80}")
        print(f"🔄 Quantizing with embedding type: {embedding_type}")
        print(f"📥 Input:  {self.input_model}")
        print(f"📤 Output: {output_file}")
        print(f"💻 Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=600  # 10 minute timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                # Get output file size
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                
                print(f"✅ Success! Duration: {duration:.2f}s, Size: {file_size_mb:.2f} MB")
                
                # Record newly created file
                if not file_already_existed:
                    self.newly_created_files.add(output_file)
                
                # Print part of output
                if result.stdout:
                    print("\n📊 Quantization output:")
                    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                
                return True
            else:
                print(f"❌ Failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout (exceeded 10 minutes)")
            return False
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def benchmark_model(self, output_suffix):
        """
        Benchmark model
        
        Args:
            output_suffix: Output file suffix (lowercase format, e.g., q6_k)
        
        Returns:
            dict: Dictionary with benchmark results, or None if failed
        """
        model_file = self.output_dir / f"ggml-model-i2_s-embed-{output_suffix}.gguf"
        
        if not model_file.exists():
            print(f"❌ Model file not found for benchmarking: {model_file}")
            return None
        
        cmd = [
            str(self.bench_bin),
            "-m", str(model_file),
            "-p", "128",
            "-n", "0",
            "-t", "1,2,4,8",
            "-ngl", "0"
        ]
        
        print(f"\n{'='*80}")
        print(f"🏃 Running benchmark for: {output_suffix}")
        print(f"💻 Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Benchmark completed successfully")
                print("\n📊 Benchmark output:")
                print(result.stdout)
                
                # 解析输出
                bench_results = self.parse_benchmark_output(result.stdout, output_suffix)
                return bench_results
            else:
                print(f"❌ Benchmark failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark timeout (exceeded 5 minutes)")
            return None
            
        except Exception as e:
            print(f"❌ Benchmark exception: {e}")
            return None
    
    def parse_benchmark_output(self, output, output_suffix):
        """
        Parse benchmark output to extract t/s data (mean±std)
        
        Args:
            output: Benchmark command output
            output_suffix: Output file suffix
        
        Returns:
            dict: Dictionary with parsed results
        """
        results = {
            'embedding_type': output_suffix,
            'threads_1': None,
            'threads_2': None,
            'threads_4': None,
            'threads_8': None,
        }
        
        # Parse table data
        # Find lines containing pp128 and t/s
        lines = output.strip().split('\n')
        
        for line in lines:
            # Skip header and separator lines
            if '|' not in line or 'model' in line or '---' in line:
                continue
            
            # Try to extract data
            # Format similar to: | bitnet-25 2B I2_S - 2 bpw ternary | 1012.28 MiB |     2.74 B | CPU        |      12 |         pp128 |        405.73 ± 3.69 |
            parts = [p.strip() for p in line.split('|')]
            
            if len(parts) >= 8 and 'pp128' in parts[6]:
                threads_str = parts[5].strip()
                throughput_str = parts[7].strip()
                
                # Extract thread count
                try:
                    threads = int(threads_str)
                except:
                    continue
                
                # Extract t/s data (format: "405.73 ± 3.69" or "405.73")
                # Try to match "mean ± std" format
                match_with_std = re.search(r'([\d.]+)\s*±\s*([\d.]+)', throughput_str)
                if match_with_std:
                    mean = float(match_with_std.group(1))
                    std = float(match_with_std.group(2))
                    throughput = f"{mean:.2f}±{std:.2f}"
                else:
                    # Only mean, no std
                    match = re.search(r'([\d.]+)', throughput_str)
                    if match:
                        throughput = f"{float(match.group(1)):.2f}"
                    else:
                        continue
                
                # Store result based on thread count
                if threads == 1:
                    results['threads_1'] = throughput
                elif threads == 2:
                    results['threads_2'] = throughput
                elif threads == 4:
                    results['threads_4'] = throughput
                elif threads == 8:
                    results['threads_8'] = throughput
        
        return results
    
    def cleanup_model(self, output_suffix):
        """
        Cleanup model files (only delete newly created files)
        
        Args:
            output_suffix: Output file suffix
        """
        model_file = self.output_dir / f"ggml-model-i2_s-embed-{output_suffix}.gguf"
        
        if model_file in self.newly_created_files:
            try:
                model_file.unlink()
                print(f"🗑️  Deleted newly created file: {model_file}")
                self.newly_created_files.remove(model_file)
            except Exception as e:
                print(f"⚠️  Failed to delete {model_file}: {e}")
        else:
            print(f"ℹ️  Keeping existing file: {model_file}")
    
    def run_all_quantizations(self, types_to_quantize):
        """
        Run all quantizations
        
        Args:
            types_to_quantize: List of quantization types, tuples of (embedding_type, output_suffix)
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting Embedding Quantization and Benchmarking")
        print(f"{'='*80}")
        print(f"📥 Input model: {self.input_model}")
        print(f"📤 Output directory: {self.output_dir}")
        print(f"📊 Stats directory: {self.stats_dir}")
        print(f"🔢 Total quantizations: {len(types_to_quantize)}")
        print(f"{'='*80}\n")
        
        total_start = datetime.now()
        
        for i, (embedding_type, output_suffix) in enumerate(types_to_quantize, 1):
            print(f"\n{'#'*80}")
            print(f"[{i}/{len(types_to_quantize)}] Processing {output_suffix} ({embedding_type})")
            print(f"{'#'*80}\n")
            
            # Quantize model
            success = self.quantize(embedding_type, output_suffix)
            
            if not success:
                print(f"⚠️  Skipping benchmark for {output_suffix} due to quantization failure")
                continue
            
            # Run benchmark
            bench_results = self.benchmark_model(output_suffix)
            
            if bench_results:
                self.results.append(bench_results)
            else:
                print(f"⚠️  Benchmark failed for {output_suffix}")
            
            # Cleanup model files (only delete newly created files)
            self.cleanup_model(output_suffix)
            
            print(f"\n{'#'*80}")
            print(f"✅ Completed {output_suffix}")
            print(f"{'#'*80}\n")
        
        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds()
        
        # 保存结果到CSV
        self.save_results_to_csv()
        
        # 打印总结
        self.print_summary(total_duration)
    
    def save_results_to_csv(self):
        """将benchmark结果保存到CSV文件"""
        if not self.results:
            print("⚠️  No results to save")
            return
        
        # Use user-specified CSV path, otherwise use default path
        if self.csv_output:
            csv_file = self.csv_output
            # Ensure parent directory exists
            csv_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            csv_file = self.stats_dir / f"embedding_benchmark.csv"
        
        print(f"\n💾 Saving results to: {csv_file}")
        
        try:
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ['embedding_type', 'threads_1', 'threads_2', 'threads_4', 'threads_8']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result)
            
            print(f"✅ Results saved successfully")
            
            # Also print table
            print(f"\n📊 Benchmark Results:")
            print(f"{'Type':<15} {'1 thread':<18} {'2 threads':<18} {'4 threads':<18} {'8 threads':<18}")
            print("-" * 87)
            for result in self.results:
                t1 = result['threads_1'] if result['threads_1'] else "N/A"
                t2 = result['threads_2'] if result['threads_2'] else "N/A"
                t4 = result['threads_4'] if result['threads_4'] else "N/A"
                t8 = result['threads_8'] if result['threads_8'] else "N/A"
                print(f"{result['embedding_type']:<15} {t1:<18} {t2:<18} {t4:<18} {t8:<18}")
                
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
        
    def print_summary(self, total_duration):
        """Print quantization summary"""
        print(f"\n\n{'='*80}")
        print(f"📊 QUANTIZATION AND BENCHMARK SUMMARY")
        print(f"{'='*80}\n")
        
        successful = len(self.results)
        total = len(self.results)
        
        print(f"✅ Completed: {successful} benchmarks")
        print(f"⏱️  Total duration: {total_duration/60:.2f} minutes\n")
        
        if self.results:
            if self.csv_output and self.csv_output.exists():
                print(f"📁 Results saved to: {self.csv_output}")
            else:
                csv_files = list(self.stats_dir.glob("embedding_benchmark*.csv"))
                if csv_files:
                    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                    print(f"📁 Results saved to: {latest_csv}")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Quantize model embeddings to multiple formats')
    parser.add_argument('--input', '-i',
                        default='../models/BitNet-b1.58-2B-4T/ggml-model-f32.gguf',
                        help='Input model path (default: ../models/BitNet-b1.58-2B-4T/ggml-model-f32.gguf)')
    parser.add_argument('--output-dir', '-o',
                        default='../models/BitNet-b1.58-2B-4T',
                        help='Output directory (default: ../models/BitNet-b1.58-2B-4T)')
    parser.add_argument('--quantize-bin', '-q',
                        default='../build/bin/llama-quantize',
                        help='Path to llama-quantize binary (default: ../build/bin/llama-quantize)')
    parser.add_argument('--bench-bin', '-b',
                        default='../build/bin/llama-bench',
                        help='Path to llama-bench binary (default: ../build/bin/llama-bench)')
    parser.add_argument('--stats-dir',
                        default='../stats',
                        help='Directory to save benchmark results (default: ../stats)')
    parser.add_argument('--csv-output', '-c',
                        help='Custom path for CSV output file (e.g., stats/my_results.csv)')
    parser.add_argument('--types', '-t',
                        nargs='+',
                        help='Specific types to quantize (e.g., f32 q6_k q4_0)')
    parser.add_argument('--skip-existing', '-s',
                        action='store_true',
                        help='Skip quantization if output file already exists (will still benchmark existing files)')
    
    args = parser.parse_args()
    
    # Define all supported quantization types
    # Format: (embedding_type for command line, output_suffix for filename)
    all_types = [
        ('F32', 'f32'),
        ('F16', 'f16'),
        ('Q8_0', 'q8_0'),
        ('Q6_K', 'q6_k'),
        ('Q5_0', 'q5_0'),
        ('Q4_0', 'q4_0'),
        ('Q3_K', 'q3_k'),
        ('TQ2_0', 'tq2_0'),
    ]
    
    # If specific types are specified, filter the list
    if args.types:
        types_lower = [t.lower() for t in args.types]
        types_to_quantize = [(et, os) for et, os in all_types if os.lower() in types_lower]
        if not types_to_quantize:
            print(f"❌ No valid types specified. Available types: {', '.join([os for _, os in all_types])}")
            return
    else:
        types_to_quantize = all_types
    
    # If skip existing files is enabled, no need to filter
    # Because new logic will automatically detect and skip during quantization, but will still benchmark
    
    # 创建量化器并运行
    try:
        quantizer = EmbeddingQuantizer(
            args.input, 
            args.output_dir, 
            args.quantize_bin,
            args.bench_bin,
            args.stats_dir,
            args.csv_output
        )
        quantizer.run_all_quantizations(types_to_quantize)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Quantization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main() or 0)
