#!/usr/bin/env python3
"""
Comprehensive stress test script for dyxgb.

This script runs dyxgb through a battery of tests covering:
- Different data sources (CSV, Parquet, JSON, databases)
- Different task types (classification, regression, multi-class)
- Different dataset sizes (small, medium, large, xlarge)
- Edge cases (missing values, outliers, imbalanced classes)
- Transform pipelines
- Hyperparameter tuning
- All CLI commands (train, predict, evaluate, importance)
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple
import json


class Color:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print a header with formatting."""
    print(f"\n{Color.BOLD}{Color.HEADER}{'='*70}{Color.END}")
    print(f"{Color.BOLD}{Color.HEADER}{text.center(70)}{Color.END}")
    print(f"{Color.BOLD}{Color.HEADER}{'='*70}{Color.END}\n")


def print_test(test_name: str) -> None:
    """Print test name."""
    print(f"{Color.CYAN}▶ Running: {Color.BOLD}{test_name}{Color.END}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Color.GREEN}✓ {message}{Color.END}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"{Color.RED}✗ {message}{Color.END}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{Color.YELLOW}⚠ {message}{Color.END}")


def run_command(cmd: List[str], timeout: int = 300) -> Tuple[bool, str, float]:
    """
    Run a command and return success status, output, and execution time.
    
    Args:
        cmd: Command to run as list of strings
        timeout: Timeout in seconds
    
    Returns:
        Tuple of (success, output, execution_time)
    """
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/home/runner/work/dyxgb/dyxgb"
        )
        execution_time = time.time() - start_time
        
        # Check if command succeeded
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        return success, output, execution_time
    
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return False, f"Command timed out after {timeout}s", execution_time
    
    except Exception as e:
        execution_time = time.time() - start_time
        return False, f"Exception: {str(e)}", execution_time


class TestResult:
    """Store test results."""
    def __init__(self, name: str, success: bool, time: float, error: str = ""):
        self.name = name
        self.success = success
        self.time = time
        self.error = error


class StressTest:
    """Main stress test runner."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.base_path = Path("/home/runner/work/dyxgb/dyxgb")
        self.test_data_path = self.base_path / "test_data"
        self.models_path = self.test_data_path / "models"
        self.output_path = self.test_data_path / "output"
        
        # Ensure output directories exist
        self.models_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
    
    def add_result(self, name: str, success: bool, time: float, error: str = "") -> None:
        """Add a test result."""
        self.results.append(TestResult(name, success, time, error))
        
        if success:
            print_success(f"Completed in {time:.2f}s")
        else:
            print_error(f"Failed in {time:.2f}s: {error}")
    
    def test_basic_classification_csv(self) -> None:
        """Test 1: Basic classification with CSV file."""
        print_test("Test 1: Basic classification (CSV, small dataset)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "test_data/classification_small.csv",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test1_model.json",
            "--encoder-output", "test_data/models/test1_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Basic CSV Classification", success, exec_time, output if not success else "")
    
    def test_parquet_medium(self) -> None:
        """Test 2: Medium dataset with Parquet format."""
        print_test("Test 2: Classification (Parquet, medium dataset)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "test_data/classification_medium.parquet",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test2_model.json",
            "--encoder-output", "test_data/models/test2_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Parquet Medium Classification", success, exec_time, output if not success else "")
    
    def test_json_format(self) -> None:
        """Test 3: JSON file format."""
        print_test("Test 3: Classification (JSON format)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "test_data/classification_small.json",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test3_model.json",
            "--encoder-output", "test_data/models/test3_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("JSON Format Classification", success, exec_time, output if not success else "")
    
    def test_regression(self) -> None:
        """Test 4: Regression task."""
        print_test("Test 4: Regression task")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "test_data/regression_medium.csv",
            "--target", "price",
            "--task", "regression",
            "--output", "test_data/models/test4_model.json"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Regression Task", success, exec_time, output if not success else "")
    
    def test_large_dataset(self) -> None:
        """Test 5: Large dataset (10k rows)."""
        print_test("Test 5: Large dataset stress test (10k rows)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "test_data/classification_large.csv",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test5_model.json",
            "--encoder-output", "test_data/models/test5_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd, timeout=600)
        self.add_result("Large Dataset (10k)", success, exec_time, output if not success else "")
    
    def test_xlarge_dataset(self) -> None:
        """Test 6: Extra large dataset (50k rows)."""
        print_test("Test 6: Extra large dataset stress test (50k rows)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "test_data/classification_xlarge.parquet",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test6_model.json",
            "--encoder-output", "test_data/models/test6_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd, timeout=900)
        self.add_result("Extra Large Dataset (50k)", success, exec_time, output if not success else "")
    
    def test_multiclass(self) -> None:
        """Test 7: Multi-class classification."""
        print_test("Test 7: Multi-class classification (5 classes)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "test_data/multiclass_classification.parquet",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test7_model.json",
            "--encoder-output", "test_data/models/test7_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Multi-class Classification", success, exec_time, output if not success else "")
    
    def test_config_basic(self) -> None:
        """Test 8: Config file - basic classification."""
        print_test("Test 8: Config file (basic classification)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--config", "test_data/config_classification_basic.yaml"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Config Basic Classification", success, exec_time, output if not success else "")
    
    def test_config_advanced(self) -> None:
        """Test 9: Config file - advanced with transforms."""
        print_test("Test 9: Config file (advanced with transforms)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--config", "test_data/config_classification_advanced.yaml"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Config Advanced with Transforms", success, exec_time, output if not success else "")
    
    def test_config_regression(self) -> None:
        """Test 10: Config file - regression."""
        print_test("Test 10: Config file (regression)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--config", "test_data/config_regression.yaml"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Config Regression", success, exec_time, output if not success else "")
    
    def test_prediction(self) -> None:
        """Test 11: Prediction command."""
        print_test("Test 11: Prediction command")
        
        # First train a model if not exists
        if not (self.models_path / "test1_model.json").exists():
            self.test_basic_classification_csv()
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "predict",
            "--source", "test_data/classification_test.csv",
            "--model", "test_data/models/test1_model.json",
            "--encoder", "test_data/models/test1_encoder.joblib",
            "--task", "classification",
            "--output", "test_data/output/test11_predictions.parquet"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Prediction Command", success, exec_time, output if not success else "")
    
    def test_evaluation(self) -> None:
        """Test 12: Evaluation command."""
        print_test("Test 12: Evaluation command")
        
        # Ensure model exists
        if not (self.models_path / "test1_model.json").exists():
            self.test_basic_classification_csv()
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "evaluate",
            "--source", "test_data/classification_test.csv",
            "--model", "test_data/models/test1_model.json",
            "--encoder", "test_data/models/test1_encoder.joblib",
            "--target", "label",
            "--task", "classification"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Evaluation Command", success, exec_time, output if not success else "")
    
    def test_importance(self) -> None:
        """Test 13: Feature importance command."""
        print_test("Test 13: Feature importance command")
        
        # Ensure model exists
        if not (self.models_path / "test1_model.json").exists():
            self.test_basic_classification_csv()
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "importance",
            "--model", "test_data/models/test1_model.json",
            "--task", "classification",
            "--top", "10",
            "--output", "test_data/output/test13_importance.json"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Feature Importance", success, exec_time, output if not success else "")
    
    def test_edge_cases(self) -> None:
        """Test 14: Edge case dataset."""
        print_test("Test 14: Edge case dataset (extreme values, high missing data)")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "test_data/edge_cases.csv",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test14_model.json",
            "--encoder-output", "test_data/models/test14_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("Edge Cases Dataset", success, exec_time, output if not success else "")
    
    def test_database_sqlite(self) -> None:
        """Test 15: SQLite database source."""
        print_test("Test 15: SQLite database source")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "sqlite:///test_data/test_data.sqlite",
            "--table", "train_data",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test15_model.json",
            "--encoder-output", "test_data/models/test15_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("SQLite Database", success, exec_time, output if not success else "")
    
    def test_database_duckdb(self) -> None:
        """Test 16: DuckDB database source."""
        print_test("Test 16: DuckDB database source")
        
        cmd = [
            "python3", "-m", "dyxgb.cli", "train",
            "--source", "duckdb:///test_data/test_data.duckdb",
            "--table", "train_data",
            "--target", "label",
            "--task", "classification",
            "--output", "test_data/models/test16_model.json",
            "--encoder-output", "test_data/models/test16_encoder.joblib"
        ]
        
        success, output, exec_time = run_command(cmd)
        self.add_result("DuckDB Database", success, exec_time, output if not success else "")
    
    def run_all_tests(self) -> None:
        """Run all stress tests."""
        print_header("DYXGB COMPREHENSIVE STRESS TEST")
        
        print(f"{Color.BLUE}Test Data Location: {self.test_data_path}{Color.END}")
        print(f"{Color.BLUE}Models Output: {self.models_path}{Color.END}")
        print(f"{Color.BLUE}Predictions Output: {self.output_path}{Color.END}\n")
        
        # Install dependencies
        print_test("Installing dependencies")
        cmd = ["python3", "-m", "pip", "install", "-e", ".", "--quiet"]
        success, output, exec_time = run_command(cmd, timeout=120)
        if success:
            print_success(f"Dependencies installed in {exec_time:.2f}s")
        else:
            print_warning("Dependency installation had issues, continuing anyway...")
        
        # Run all tests
        tests = [
            self.test_basic_classification_csv,
            self.test_parquet_medium,
            self.test_json_format,
            self.test_regression,
            self.test_large_dataset,
            self.test_multiclass,
            self.test_config_basic,
            self.test_config_advanced,
            self.test_config_regression,
            self.test_prediction,
            self.test_evaluation,
            self.test_importance,
            self.test_edge_cases,
            self.test_database_sqlite,
            self.test_database_duckdb,
        ]
        
        # Note: Skipping xlarge test by default (too slow), but available
        # tests.append(self.test_xlarge_dataset)
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print_error(f"Test crashed: {str(e)}")
                self.add_result(test.__name__, False, 0, str(e))
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print test summary."""
        print_header("TEST SUMMARY")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        total_time = sum(r.time for r in self.results)
        
        print(f"{Color.BOLD}Total Tests: {total}{Color.END}")
        print(f"{Color.GREEN}Passed: {passed}{Color.END}")
        print(f"{Color.RED}Failed: {failed}{Color.END}")
        print(f"{Color.CYAN}Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes){Color.END}\n")
        
        # Show individual results
        print(f"{Color.BOLD}Individual Results:{Color.END}\n")
        for i, result in enumerate(self.results, 1):
            status = f"{Color.GREEN}✓ PASS{Color.END}" if result.success else f"{Color.RED}✗ FAIL{Color.END}"
            print(f"{i:2d}. {status} | {result.name:40s} | {result.time:6.2f}s")
            if not result.success and result.error:
                # Show first line of error
                first_error = result.error.split('\n')[0][:100]
                print(f"    {Color.RED}Error: {first_error}{Color.END}")
        
        # Save results to file
        results_file = self.test_data_path / "stress_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'total_time': total_time
                },
                'results': [
                    {
                        'name': r.name,
                        'success': r.success,
                        'time': r.time,
                        'error': r.error
                    }
                    for r in self.results
                ]
            }, f, indent=2)
        
        print(f"\n{Color.CYAN}Results saved to: {results_file}{Color.END}")
        
        # Exit with appropriate code
        sys.exit(0 if failed == 0 else 1)


def main():
    """Main entry point."""
    test_runner = StressTest()
    test_runner.run_all_tests()


if __name__ == "__main__":
    main()
