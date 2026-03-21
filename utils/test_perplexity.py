#!/usr/bin/env python3
"""
Perplexity Test Script
Tests GGUF model perplexity on multiple datasets using llama-perplexity.
"""

import os
import subprocess
import time
import csv
import re
from datetime import datetime
from pathlib import Path
import argparse
import tempfile
import shutil
import statistics


class PerplexityTester:
    def __init__(self, model_path, llama_perplexity_bin="../build/bin/llama-perplexity", 
                 data_dir="../data", output_dir="perplexity_results", quick_mode=False,
                 quantize_bin="../build/bin/llama-quantize", test_embeddings=False, csv_output=None):
        self.model_path = Path(model_path)
        self.llama_perplexity_bin = Path(llama_perplexity_bin)
        self.quantize_bin = Path(quantize_bin)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.quick_mode = quick_mode
        self.test_embeddings = test_embeddings
        self.csv_output = Path(csv_output) if csv_output else None
        self.results = []
        self.created_models = set()  # Track newly created model files
        self.temp_files = []  # Track temporary files for cleanup
        
        # Embedding types to test
        self.embedding_types = [
            ('F32', 'f32'),
            ('F16', 'f16'),
            ('Q8_0', 'q8_0'),
            ('Q6_K', 'q6_k'),
            ('Q5_0', 'q5_0'),
            ('Q4_0', 'q4_0'),
            ('Q3_K', 'q3_k'),
            ('TQ2_0', 'tq2_0'),
        ]
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify llama-perplexity binary exists
        if not self.llama_perplexity_bin.exists():
            raise FileNotFoundError(f"llama-perplexity binary not found: {self.llama_perplexity_bin}")
        
        # Verify quantize binary exists if testing embeddings
        if self.test_embeddings and not self.quantize_bin.exists():
            raise FileNotFoundError(f"llama-quantize binary not found: {self.quantize_bin}")
        
        # Verify model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def find_datasets(self):
        """Find all test.txt files in dataset directories."""
        datasets = []
        
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return datasets
        
        print(f"\n🔍 Searching for datasets in {self.data_dir}...")
        
        # Look for test.txt files in subdirectories
        for dataset_dir in sorted(self.data_dir.iterdir()):
            if dataset_dir.is_dir():
                test_file = dataset_dir / "test.txt"
                if test_file.exists():
                    size_mb = test_file.stat().st_size / (1024 * 1024)
                    datasets.append({
                        'name': dataset_dir.name,
                        'path': test_file,
                        'size': test_file.stat().st_size,
                        'size_mb': size_mb
                    })
                    print(f"   ✅ {dataset_dir.name:<20} ({size_mb:.2f} MB)")
                else:
                    print(f"   ⚠️  {dataset_dir.name:<20} (no test.txt found)")
        
        return datasets
    
    def create_quick_dataset(self, dataset_path, num_chars=4096):
        """Create a temporary dataset with only the first N characters for quick testing."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        self.temp_files.append(temp_file.name)
        
        try:
            with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(num_chars)
                temp_file.write(content)
            temp_file.close()
            return Path(temp_file.name)
        except Exception as e:
            print(f"⚠️  Failed to create quick dataset: {e}")
            temp_file.close()
            return dataset_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        self.temp_files = []
    
    def run_perplexity_test(self, dataset_name, dataset_path, threads=16, ctx_size=512, model_override=None):
        """Run perplexity test on a single dataset."""
        test_model = model_override if model_override else self.model_path
        
        print(f"\n{'='*80}")
        print(f"📊 Testing on dataset: {dataset_name}")
        print(f"   File: {dataset_path}")
        print(f"   Model: {test_model.name}")
        print(f"{'='*80}")
        
        cmd = [
            str(self.llama_perplexity_bin),
            "-m", str(test_model),
            "-f", str(dataset_path),
            "-t", str(threads),
            "-c", str(ctx_size),
            "-ngl", "0"  # CPU only
        ]
        
        print(f"💻 Command: {' '.join(cmd)}")
        print(f"⏱️  Starting test...\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=os.getcwd()
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse perplexity from output (check both stdout and stderr)
                combined_output = result.stdout + "\n" + result.stderr
                ppl = self.parse_perplexity(combined_output)
                
                if ppl is not None:
                    print(f"\n✅ Perplexity: {ppl}")
                    print(f"⏱️  Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
                    status = "success"
                else:
                    print(f"\n⚠️  Test completed but could not parse perplexity")
                    print(f"Last 500 chars of stdout:")
                    print(result.stdout[-500:])
                    print(f"Last 500 chars of stderr:")
                    print(result.stderr[-500:])
                    status = "parse_error"
                    ppl = None
            else:
                print(f"\n❌ Test failed with return code {result.returncode}")
                print(f"Error: {result.stderr[:500]}")
                status = "failed"
                ppl = None
                elapsed_time = time.time() - start_time
            
            return {
                'dataset': dataset_name,
                'perplexity': ppl,
                'time': elapsed_time,
                'status': status,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            print(f"\n❌ Timeout after {elapsed_time:.2f}s")
            return {
                'dataset': dataset_name,
                'perplexity': None,
                'time': elapsed_time,
                'status': 'timeout',
                'stdout': '',
                'stderr': 'Test exceeded 1 hour timeout'
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n❌ Error: {e}")
            return {
                'dataset': dataset_name,
                'perplexity': None,
                'time': elapsed_time,
                'status': 'error',
                'stdout': '',
                'stderr': str(e)
            }
    
    def parse_perplexity(self, output):
        """Parse perplexity value (mean±std format) from llama-perplexity output."""
        # First try to match "PPL = mean +/- std" format
        pattern_with_std = r'PPL\s*=\s*(\d+\.?\d*)\s*\+/-\s*(\d+\.?\d*)'
        match = re.search(pattern_with_std, output, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                mean = float(match.group(1))
                std = float(match.group(2))
                return f"{mean:.4f}±{std:.4f}"
            except ValueError:
                pass
        
        # Fallback to patterns without std
        patterns = [
            r'Final estimate:\s*PPL\s*=\s*(\d+\.?\d*)',
            r'Final perplexity:\s*(\d+\.?\d*)',
            r'PPL\s*=\s*(\d+\.?\d*)',
            r'PPL:\s*(\d+\.?\d*)',
            r'perplexity:\s*(\d+\.?\d*)',
            r'ppl\s*=\s*(\d+\.?\d*)',
            r'Perplexity:\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return f"{float(match.group(1)):.4f}"
                except ValueError:
                    continue
        
        return None
    
    def quantize_embedding(self, embedding_type, output_suffix):
        """
        Quantize model with specific embedding type.
        
        Args:
            embedding_type: Token embedding type (uppercase, e.g., 'Q6_K')
            output_suffix: Output file suffix (lowercase, e.g., 'q6_k')
        
        Returns:
            Path to quantized model or None if failed
        """
        # Construct output path
        model_dir = self.model_path.parent
        output_path = model_dir / f"ggml-model-i2_s-embed-{output_suffix}.gguf"
        
        # Check if file already exists
        file_existed = output_path.exists()
        
        if file_existed:
            print(f"ℹ️  Model already exists: {output_path.name}")
            return output_path
        
        cmd = [
            str(self.quantize_bin),
            "--token-embedding-type", embedding_type,
            str(self.model_path),
            str(output_path),
            "I2_S",
            "1",
            "1"
        ]
        
        print(f"\n{'='*80}")
        print(f"🔄 Quantizing with embedding type: {embedding_type}")
        print(f"📥 Input:  {self.model_path.name}")
        print(f"📤 Output: {output_path.name}")
        print(f"💻 Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=600  # 10 minutes timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✅ Quantization successful!")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Size: {file_size_mb:.2f} MB")
                
                # Mark as newly created
                self.created_models.add(output_path)
                return output_path
            else:
                print(f"❌ Quantization failed with return code {result.returncode}")
                print(f"Error: {result.stderr[:500]}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Quantization timeout (exceeded 10 minutes)")
            return None
        except Exception as e:
            print(f"❌ Quantization error: {e}")
            return None
    
    def cleanup_model(self, model_path):
        """Delete model file if it was created during this session."""
        if model_path in self.created_models:
            try:
                model_path.unlink()
                print(f"🗑️  Deleted: {model_path.name}")
                self.created_models.remove(model_path)
            except Exception as e:
                print(f"⚠️  Failed to delete {model_path.name}: {e}")
        else:
            print(f"ℹ️  Keeping existing file: {model_path.name}")
    
    def run_all_tests(self, threads=16, ctx_size=512):
        """Run perplexity tests on all datasets."""
        datasets = self.find_datasets()
        
        if not datasets:
            print(f"\n❌ No datasets found in {self.data_dir}")
            print(f"   Make sure each dataset directory has a test.txt file")
            return
        
        # Quick mode: test all datasets but only first 4096 chars with smaller context
        if self.quick_mode:
            ctx_size = min(ctx_size, 128)  # Use smaller context in quick mode
            print(f"\n⚡ QUICK TEST MODE ENABLED")
            print(f"   - Testing all datasets with first 4096 characters only")
            print(f"   - Using reduced context size: {ctx_size}")
        
        # Determine models to test
        if self.test_embeddings:
            print(f"\n{'='*80}")
            print(f"🧪 EMBEDDING QUANTIZATION TEST MODE")
            print(f"{'='*80}")
            print(f"📦 Base model: {self.model_path.name}")
            print(f"🔢 Embedding types to test: {len(self.embedding_types)}")
            print(f"📊 Datasets: {len(datasets)}")
            print(f"🧵 Threads: {threads}")
            print(f"📏 Context size: {ctx_size}")
            print(f"{'='*80}")
            
            total_start = time.time()
            
            # Test each embedding type
            for i, (embedding_type, output_suffix) in enumerate(self.embedding_types, 1):
                print(f"\n\n{'#'*80}")
                print(f"[{i}/{len(self.embedding_types)}] Testing embedding type: {output_suffix} ({embedding_type})")
                print(f"{'#'*80}")
                
                # Quantize model
                quantized_model = self.quantize_embedding(embedding_type, output_suffix)
                
                if quantized_model is None:
                    print(f"⚠️  Skipping tests for {output_suffix} due to quantization failure")
                    continue
                
                # Test on all datasets
                for j, dataset in enumerate(datasets, 1):
                    print(f"\n[{j}/{len(datasets)}] Testing {dataset['name']} with {output_suffix}...")
                    
                    # Use quick dataset if in quick mode
                    test_path = dataset['path']
                    if self.quick_mode:
                        test_path = self.create_quick_dataset(dataset['path'])
                    
                    result = self.run_perplexity_test(
                        f"{dataset['name']}_embed-{output_suffix}",
                        test_path,
                        threads,
                        ctx_size,
                        model_override=quantized_model
                    )
                    self.results.append(result)
                
                # Cleanup model after testing
                print(f"\n🧹 Cleaning up {output_suffix} model...")
                self.cleanup_model(quantized_model)
                
                print(f"\n{'#'*80}")
                print(f"✅ Completed {output_suffix}")
                print(f"{'#'*80}")
            
            total_time = time.time() - total_start
            
        else:
            # Regular single model test
            print(f"\n{'='*80}")
            print(f"🚀 PERPLEXITY TEST SESSION{' (QUICK MODE)' if self.quick_mode else ''}")
            print(f"{'='*80}")
            print(f"📦 Model: {self.model_path.name}")
            print(f"📁 Model path: {self.model_path}")
            print(f"📊 Datasets {'to test' if self.quick_mode else 'found'}: {len(datasets)}")
            print(f"🧵 Threads: {threads}")
            print(f"📏 Context size: {ctx_size}")
            print(f"{'='*80}")
            
            total_start = time.time()
            
            # Run tests
            for i, dataset in enumerate(datasets, 1):
                print(f"\n\n[{i}/{len(datasets)}] Processing {dataset['name']}...")
                
                # Use quick dataset if in quick mode
                test_path = dataset['path']
                if self.quick_mode:
                    test_path = self.create_quick_dataset(dataset['path'])
                
                result = self.run_perplexity_test(
                    dataset['name'],
                    test_path,
                    threads,
                    ctx_size
                )
                self.results.append(result)
            
            total_time = time.time() - total_start
        
        # Clean up temporary files
        if self.quick_mode:
            print(f"\n🧹 Cleaning up temporary files...")
            self.cleanup_temp_files()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary(total_time)
    
    def save_results(self):
        """Save results to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_path.stem
        
        # Use custom CSV path if provided
        if self.csv_output:
            csv_file = self.csv_output
            # Create parent directory if needed
            csv_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            csv_file = self.output_dir / f"ppl_{model_name}_{timestamp}.csv"
        
        print(f"\n💾 Saving results...")
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['dataset', 'perplexity', 'time_seconds', 'status'])
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'dataset': result['dataset'],
                    'perplexity': result['perplexity'] if result['perplexity'] is not None else 'N/A',
                    'time_seconds': f"{result['time']:.2f}",
                    'status': result['status']
                })
        
        print(f"   ✅ CSV saved: {csv_file}")
        
        # Save detailed log
        log_file = self.output_dir / f"ppl_{model_name}_{timestamp}.log"
        with open(log_file, 'w') as f:
            f.write(f"Perplexity Test Results\n")
            f.write(f"{'='*80}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            
            for result in self.results:
                f.write(f"\n{'='*80}\n")
                f.write(f"Dataset: {result['dataset']}\n")
                f.write(f"Perplexity: {result['perplexity']}\n")
                f.write(f"Time: {result['time']:.2f}s\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"\nOutput:\n{result['stdout']}\n")
                if result['stderr']:
                    f.write(f"\nErrors:\n{result['stderr']}\n")
        
        print(f"   ✅ Log saved: {log_file}")
    
    def print_summary(self, total_time):
        """Print summary of all tests."""
        print(f"\n\n{'='*80}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*80}\n")
        
        # Sort results by perplexity (lower is better)
        successful = [r for r in self.results if r['perplexity'] is not None]
        failed = [r for r in self.results if r['perplexity'] is None]
        
        if successful:
            # Extract numeric value from "mean±std" format for sorting
            def get_ppl_value(result):
                ppl = result['perplexity']
                if isinstance(ppl, str) and '±' in ppl:
                    return float(ppl.split('±')[0])
                elif isinstance(ppl, str):
                    try:
                        return float(ppl)
                    except ValueError:
                        return float('inf')
                return ppl
            
            successful_sorted = sorted(successful, key=get_ppl_value)
            
            print(f"{'Dataset':<20} {'Perplexity':>20} {'Time (s)':>12} {'Status':<15}")
            print(f"{'-'*80}")
            
            for result in successful_sorted:
                ppl_str = str(result['perplexity']) if result['perplexity'] is not None else 'N/A'
                print(f"{result['dataset']:<20} {ppl_str:>20} "
                      f"{result['time']:>12.2f} {result['status']:<15}")
            
            best_ppl = str(successful_sorted[0]['perplexity'])
            print(f"\n🏆 Best result: {successful_sorted[0]['dataset']} "
                  f"(PPL: {best_ppl})")
        
        if failed:
            print(f"\n❌ Failed tests ({len(failed)}):")
            for result in failed:
                print(f"   - {result['dataset']}: {result['status']}")
        
        print(f"\n{'='*80}")
        print(f"✅ Completed: {len(successful)}/{len(self.results)}")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Test model perplexity on multiple datasets')
    parser.add_argument('--model', '-m',
                        required=True,
                        help='Path to GGUF model file')
    parser.add_argument('--data-dir', '-d',
                        default='data',
                        help='Directory containing dataset folders (default: data)')
    parser.add_argument('--threads', '-t',
                        type=int,
                        default=16,
                        help='Number of threads (default: 16)')
    parser.add_argument('--ctx-size', '-c',
                        type=int,
                        default=512,
                        help='Context size (default: 512)')
    parser.add_argument('--output-dir', '-o',
                        default='perplexity_results',
                        help='Output directory for results (default: perplexity_results)')
    parser.add_argument('--llama-perplexity',
                        default='./build/bin/llama-perplexity',
                        help='Path to llama-perplexity binary (default: ./build/bin/llama-perplexity)')
    parser.add_argument('--quick', '-q',
                        action='store_true',
                        help='Quick test mode: test all datasets with first 4096 characters and reduced context size (128)')
    parser.add_argument('--test-embeddings', '-e',
                        action='store_true',
                        help='Test different embedding quantization types (f32, f16, q8_0, q6_k, q5_0, q4_0, q3_k, tq2_0)')
    parser.add_argument('--csv-output',
                        help='Custom path for CSV output file (e.g., results/my_ppl_results.csv)')
    parser.add_argument('--quantize-bin',
                        default='./build/bin/llama-quantize',
                        help='Path to llama-quantize binary (default: ./build/bin/llama-quantize)')
    
    args = parser.parse_args()
    
    try:
        tester = PerplexityTester(
            model_path=args.model,
            llama_perplexity_bin=args.llama_perplexity,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            quick_mode=args.quick,
            quantize_bin=args.quantize_bin,
            test_embeddings=args.test_embeddings,
            csv_output=args.csv_output
        )
        
        tester.run_all_tests(
            threads=args.threads,
            ctx_size=args.ctx_size
        )
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
