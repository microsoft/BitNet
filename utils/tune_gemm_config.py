#!/usr/bin/env python3
"""
GEMM Configuration Tuning Script
This script automatically tunes ROW_BLOCK_SIZE, COL_BLOCK_SIZE, and PARALLEL_SIZE
to find the optimal configuration for maximum throughput (t/s).
"""

import subprocess
import os
import re
import csv
import shutil
from datetime import datetime
from pathlib import Path
import argparse


class GemmTuner:
    def __init__(self, config_path, model_path, threads=16):
        self.config_path = Path(config_path)
        self.model_path = model_path
        self.threads = threads
        self.backup_path = self.config_path.parent / f"gemm-config.h.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.build_dir = Path("../build")
        self.results = []
        
    def backup_config(self):
        """Backup current configuration file"""
        print(f"📦 Backing up current config to {self.backup_path}")
        shutil.copy2(self.config_path, self.backup_path)
        
    def restore_config(self):
        """Restore original configuration file"""
        print(f"♻️  Restoring original config from {self.backup_path}")
        shutil.copy2(self.backup_path, self.config_path)
        
    def generate_config(self, act_parallel, row_block_size, col_block_size, parallel_size):
        """Generate new configuration file with simplified format"""
        content = ""
        
        # Simplified configuration format
        if act_parallel:
            content += "#define ACT_PARALLEL\n"
        
        content += f"#define ROW_BLOCK_SIZE {row_block_size}\n"
        content += f"#define COL_BLOCK_SIZE {col_block_size}\n"
        content += f"#define PARALLEL_SIZE {parallel_size}\n"
        
        with open(self.config_path, 'w') as f:
            f.write(content)
    
    def rebuild_project(self):
        """Rebuild project"""
        print("🔨 Rebuilding project...")
        result = subprocess.run(
            ["cmake", "--build", str(self.build_dir), "--target", "llama-bench"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        if result.returncode != 0:
            print(f"⚠️  Build warning/error: {result.stderr}")
            return False
        return True
        
    def run_benchmark(self):
        """Run benchmark test"""
        cmd = [
            f"{self.build_dir}/bin/llama-bench",
            "-m", self.model_path,
            "-p", "128",
            "-n", "0",
            "-t", str(self.threads),
            "-ngl", "0"
        ]
        
        print(f"⚡ Running benchmark: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=300  # 5分钟超时
        )
        
        if result.returncode != 0:
            print(f"❌ Benchmark failed: {result.stderr}")
            return None
            
        return result.stdout
        
    def parse_throughput(self, output):
        """Parse pp128 throughput from output"""
        # 匹配 pp128: |         pp128 |       501.06 ± 11.37 |
        pp_pattern = r'\|\s+pp128\s+\|\s+([\d.]+)\s+±\s+([\d.]+)\s+\|'
        pp_match = re.search(pp_pattern, output)
        
        if pp_match:
            pp_throughput = float(pp_match.group(1))
            pp_std_dev = float(pp_match.group(2))
            
            return {
                'pp_throughput': pp_throughput,
                'pp_std_dev': pp_std_dev
            }
        
        return None
        
    def test_configuration(self, act_parallel, row_block_size, col_block_size, parallel_size):
        """Test single configuration"""
        config_name = f"ACT_{'ON' if act_parallel else 'OFF'}_R{row_block_size}_C{col_block_size}_P{parallel_size}"
        print(f"\n{'='*80}")
        print(f"🧪 Testing configuration: {config_name}")
        print(f"   ACT_PARALLEL: {act_parallel}")
        print(f"   ROW_BLOCK_SIZE: {row_block_size}")
        print(f"   COL_BLOCK_SIZE: {col_block_size}")
        print(f"   PARALLEL_SIZE: {parallel_size}")
        print(f"{'='*80}")
        
        # Generate configuration
        self.generate_config(act_parallel, row_block_size, col_block_size, parallel_size)
        
        # Rebuild project
        if not self.rebuild_project():
            print("⚠️  Build failed, skipping this configuration")
            return None
        
        # Run benchmark test
        output = self.run_benchmark()
        if output is None:
            return None
            
        # Parse results
        metrics = self.parse_throughput(output)
        
        if metrics is not None:
            result = {
                'act_parallel': act_parallel,
                'row_block_size': row_block_size,
                'col_block_size': col_block_size,
                'parallel_size': parallel_size,
                'config_name': config_name,
                **metrics
            }
            self.results.append(result)
            print(f"✅ PP128: {metrics['pp_throughput']:.2f} ± {metrics['pp_std_dev']:.2f} t/s")
            return result
        else:
            print("❌ Failed to parse throughput")
            return None
            
    def save_results(self, csv_path):
        """Save results to CSV file"""
        print(f"\n💾 Saving results to {csv_path}")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'config_name', 'act_parallel', 'row_block_size', 
                'col_block_size', 'parallel_size', 
                'pp_throughput', 'pp_std_dev'
            ])
            writer.writeheader()
            writer.writerows(self.results)
            
    def find_best_config(self):
        """Find the best configuration with highest throughput"""
        if not self.results:
            print("❌ No valid results found")
            return None
            
        best = max(self.results, key=lambda x: x['pp_throughput'])
        return best
        
    def run_tuning(self, configurations, output_csv=None):
        """Run test for all configurations"""
        print(f"\n🚀 Starting tuning process with {len(configurations)} configurations")
        print(f"📊 Model: {self.model_path}")
        print(f"🧵 Threads: {self.threads}\n")
        
        # Backup configuration
        self.backup_config()
        
        try:
            # Test all configurations
            for i, config in enumerate(configurations, 1):
                print(f"\n[{i}/{len(configurations)}]")
                self.test_configuration(**config)
                
            # Save results
            if output_csv is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = f"../stats/tuning_results_{timestamp}.csv"
            else:
                csv_path = output_csv
            
            # Ensure stats directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self.save_results(csv_path)
            
            # Find best configuration
            best = self.find_best_config()
            if best:
                print(f"\n{'='*80}")
                print(f"🏆 BEST CONFIGURATION FOUND!")
                print(f"{'='*80}")
                print(f"Configuration: {best['config_name']}")
                print(f"ACT_PARALLEL: {best['act_parallel']}")
                print(f"ROW_BLOCK_SIZE: {best['row_block_size']}")
                print(f"COL_BLOCK_SIZE: {best['col_block_size']}")
                print(f"PARALLEL_SIZE: {best['parallel_size']}")
                print(f"PP128 Throughput: {best['pp_throughput']:.2f} ± {best['pp_std_dev']:.2f} t/s")
                print(f"{'='*80}\n")
                
                # Show the configuration that will be written
                print("Configuration to be written to gemm-config.h:")
                print("-" * 80)
                if best['act_parallel']:
                    print("#define ACT_PARALLEL")
                print(f"#define ROW_BLOCK_SIZE {best['row_block_size']}")
                print(f"#define COL_BLOCK_SIZE {best['col_block_size']}")
                print(f"#define PARALLEL_SIZE {best['parallel_size']}")
                print("-" * 80)
                
                # Apply best configuration
                apply = input("\nDo you want to apply this configuration to gemm-config.h? (y/n): ").strip().lower()
                if apply == 'y':
                    self.generate_config(
                        best['act_parallel'],
                        best['row_block_size'],
                        best['col_block_size'],
                        best['parallel_size']
                    )
                    self.rebuild_project()
                    print("✅ Best configuration applied and project rebuilt!")
                else:
                    self.restore_config()
                    print("✅ Original configuration restored")
                
                # Clean up backup file
                if self.backup_path.exists():
                    self.backup_path.unlink()
                    print(f"🗑️  Removed backup file: {self.backup_path}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Tuning interrupted by user")
            self.restore_config()
            # Clean up backup file
            if self.backup_path.exists():
                self.backup_path.unlink()
                print(f"🗑️  Removed backup file: {self.backup_path}")
        except Exception as e:
            print(f"\n❌ Error during tuning: {e}")
            self.restore_config()
            # Clean up backup file
            if self.backup_path.exists():
                self.backup_path.unlink()
                print(f"🗑️  Removed backup file: {self.backup_path}")
            raise


def generate_configurations():
    """Generate list of configurations to test"""
    configurations = []
    
    act_parallel_options = [True]
    
    row_sizes = [2, 4, 8]#[2, 4, 8, 16, 32]
    col_sizes = [32, 64]#[32, 64, 128, 256, 512, 1024]
    parallelism_degree = [4]
    
    for act_parallel in act_parallel_options:
        for row in row_sizes:
            for col in col_sizes:
                for parallel in parallelism_degree:
                    # Add filtering conditions
                    if act_parallel:
                        # When ACT_PARALLEL=True, only calculate combinations with parallel < row
                        if parallel > row:
                            continue
                    else:
                        # When ACT_PARALLEL=False, only calculate combinations with parallel < col
                        if parallel > col:
                            continue
                    
                    configurations.append({
                        'act_parallel': act_parallel,
                        'row_block_size': row,
                        'col_block_size': col,
                        'parallel_size': parallel
                    })
    
    return configurations


def main():
    parser = argparse.ArgumentParser(description='Tune GEMM configuration for optimal performance')
    parser.add_argument('--config', default='../include/gemm-config.h', 
                        help='Path to gemm-config.h file')
    parser.add_argument('--model', default='../models/BitNet-b1.58-2B-4T/ggml-model-i2_s-embed-q6_k.gguf',
                        help='Path to model file')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads to use')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with fewer configurations')
    parser.add_argument('--custom', action='store_true',
                        help='Manually specify configurations to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (default: stats/tuning_results_<timestamp>.csv)')
    
    args = parser.parse_args()
    
    tuner = GemmTuner(args.config, args.model, args.threads)
    
    if args.custom:
        # Custom configuration mode
        print("Custom configuration mode")
        configurations = []
        while True:
            print("\nEnter configuration (or 'done' to finish):")
            act = input("ACT_PARALLEL (y/n): ").strip().lower() == 'y'
            if input == 'done':
                break
            row = int(input("ROW_BLOCK_SIZE: "))
            col = int(input("COL_BLOCK_SIZE: "))
            par = int(input("PARALLEL_SIZE: "))
            configurations.append({
                'act_parallel': act,
                'row_block_size': row,
                'col_block_size': col,
                'parallel_size': par
            })
    elif args.quick:
        # Quick test mode - test only a few key configurations
        configurations = [
            {'act_parallel': True, 'row_block_size': 4, 'col_block_size': 128, 'parallel_size': 4},
            {'act_parallel': True, 'row_block_size': 8, 'col_block_size': 128, 'parallel_size': 4},
            {'act_parallel': True, 'row_block_size': 4, 'col_block_size': 64, 'parallel_size': 4},
            {'act_parallel': False, 'row_block_size': 32, 'col_block_size': 4, 'parallel_size': 4},
            {'act_parallel': False, 'row_block_size': 16, 'col_block_size': 4, 'parallel_size': 4},
        ]
    else:
        # Full test mode
        configurations = generate_configurations()
    
    print(f"\n{'='*80}")
    print(f"GEMM Configuration Tuner")
    print(f"{'='*80}")
    print(f"Total configurations to test: {len(configurations)}")
    print(f"Estimated time: ~{len(configurations) * 0.5:.1f} minutes (assuming 30s per test)")
    print(f"{'='*80}\n")
    
    proceed = input("Proceed with tuning? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Tuning cancelled")
        return
    
    tuner.run_tuning(configurations, output_csv=args.output)


if __name__ == "__main__":
    main()
