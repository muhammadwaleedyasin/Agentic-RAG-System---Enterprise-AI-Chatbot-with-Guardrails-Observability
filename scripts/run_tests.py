#!/usr/bin/env python3
"""Test runner script for Enterprise RAG Chatbot testing suite."""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class TestRunner:
    """Comprehensive test runner for RAG system."""
    
    def __init__(self):
        """Initialize test runner."""
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_unit_tests(self, coverage: bool = True, parallel: bool = False) -> Dict:
        """Run unit tests with coverage reporting."""
        print("🧪 Running Unit Tests...")
        
        cmd = ["python", "-m", "pytest", "tests/unit/", "-v"]
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml",
                "--cov-fail-under=85"
            ])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        cmd.extend([
            "--junit-xml=test-results/unit-tests.xml",
            "-m", "unit"
        ])
        
        return self._run_command(cmd, "unit_tests")
    
    def run_integration_tests(self) -> Dict:
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest", "tests/integration/", "-v",
            "--junit-xml=test-results/integration-tests.xml",
            "-m", "integration"
        ]
        
        return self._run_command(cmd, "integration_tests")
    
    def run_evaluation_tests(self, dataset_path: Optional[str] = None) -> Dict:
        """Run RAG evaluation tests."""
        print("📊 Running RAG Evaluation Tests...")
        
        cmd = [
            "python", "-m", "pytest", "tests/evaluation/", "-v",
            "--junit-xml=test-results/evaluation-tests.xml",
            "-m", "evaluation",
            "--tb=short"
        ]
        
        if dataset_path:
            cmd.extend(["--dataset-path", dataset_path])
        
        return self._run_command(cmd, "evaluation_tests")
    
    def run_performance_tests(self, duration: int = 60) -> Dict:
        """Run performance and load tests."""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest", "tests/performance/", "-v",
            "--junit-xml=test-results/performance-tests.xml",
            "-m", "performance",
            "--durations=10",
            f"--test-duration={duration}"
        ]
        
        return self._run_command(cmd, "performance_tests")
    
    def run_benchmarks(self) -> Dict:
        """Run system benchmarks."""
        print("🏁 Running System Benchmarks...")
        
        benchmark_script = self.project_root / "tests/performance/run_benchmarks.py"
        
        if benchmark_script.exists():
            cmd = ["python", str(benchmark_script)]
            return self._run_command(cmd, "benchmarks")
        else:
            print("⚠️  Benchmark script not found, skipping benchmarks")
            return {"success": False, "message": "Benchmark script not found"}
    
    def run_security_tests(self) -> Dict:
        """Run security analysis."""
        print("🔒 Running Security Analysis...")
        
        results = {}
        
        # Run bandit for security issues
        print("  Running Bandit security scanner...")
        bandit_cmd = [
            "bandit", "-r", "src/", "-f", "json", "-o", "security-reports/bandit.json"
        ]
        bandit_result = self._run_command(bandit_cmd, "bandit", check_return_code=False)
        results["bandit"] = bandit_result
        
        # Run safety for dependency vulnerabilities
        print("  Running Safety dependency scanner...")
        safety_cmd = [
            "safety", "check", "--json", "--output", "security-reports/safety.json"
        ]
        safety_result = self._run_command(safety_cmd, "safety", check_return_code=False)
        results["safety"] = safety_result
        
        return results
    
    def run_linting(self) -> Dict:
        """Run code linting and formatting checks."""
        print("✨ Running Code Quality Checks...")
        
        results = {}
        
        # Black formatting check
        print("  Checking code formatting with Black...")
        black_cmd = ["black", "--check", "--diff", "src/", "tests/"]
        results["black"] = self._run_command(black_cmd, "black", check_return_code=False)
        
        # isort import sorting check
        print("  Checking import sorting with isort...")
        isort_cmd = ["isort", "--check-only", "--diff", "src/", "tests/"]
        results["isort"] = self._run_command(isort_cmd, "isort", check_return_code=False)
        
        # Flake8 linting
        print("  Running Flake8 linter...")
        flake8_cmd = ["flake8", "src/", "tests/", "--max-line-length=88", "--extend-ignore=E203,W503"]
        results["flake8"] = self._run_command(flake8_cmd, "flake8", check_return_code=False)
        
        # MyPy type checking
        print("  Running MyPy type checker...")
        mypy_cmd = ["mypy", "src/", "--ignore-missing-imports"]
        results["mypy"] = self._run_command(mypy_cmd, "mypy", check_return_code=False)
        
        return results
    
    def run_all_tests(self, 
                     skip_performance: bool = False,
                     skip_evaluation: bool = False,
                     coverage: bool = True) -> Dict:
        """Run all test suites."""
        print("🚀 Running Complete Test Suite...")
        
        self.start_time = datetime.now()
        
        # Create output directories
        self._create_output_directories()
        
        # Run test suites
        all_results = {}
        
        # Code quality first
        all_results["linting"] = self.run_linting()
        
        # Security analysis
        all_results["security"] = self.run_security_tests()
        
        # Unit tests
        all_results["unit_tests"] = self.run_unit_tests(coverage=coverage, parallel=True)
        
        # Integration tests
        all_results["integration_tests"] = self.run_integration_tests()
        
        # Evaluation tests (optional)
        if not skip_evaluation:
            all_results["evaluation_tests"] = self.run_evaluation_tests()
        
        # Performance tests (optional)
        if not skip_performance:
            all_results["performance_tests"] = self.run_performance_tests()
            all_results["benchmarks"] = self.run_benchmarks()
        
        self.end_time = datetime.now()
        self.test_results = all_results
        
        # Generate summary report
        self._generate_summary_report()
        
        return all_results
    
    def _run_command(self, cmd: List[str], test_type: str, check_return_code: bool = True) -> Dict:
        """Run a command and capture results."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600  # 10 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0 if check_return_code else True
            
            return {
                "success": success,
                "return_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "return_code": -1,
                "duration": 600,
                "stdout": "",
                "stderr": f"Command timed out after 10 minutes: {' '.join(cmd)}",
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "return_code": -1,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": f"Error running command: {e}",
                "command": " ".join(cmd)
            }
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        directories = [
            "test-results",
            "security-reports",
            "performance-reports",
            "evaluation-reports",
            "htmlcov"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def _generate_summary_report(self):
        """Generate comprehensive test summary report."""
        if not self.test_results:
            return
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        summary = {
            "test_run_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat()
            },
            "results": self.test_results,
            "overall_status": self._calculate_overall_status(),
            "recommendations": self._generate_recommendations()
        }
        
        # Write JSON report
        with open("test-results/summary-report.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate HTML report
        self._generate_html_report(summary)
        
        # Print summary to console
        self._print_summary_to_console(summary)
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall test status."""
        critical_tests = ["unit_tests", "integration_tests", "linting"]
        
        for test_type in critical_tests:
            if test_type in self.test_results:
                result = self.test_results[test_type]
                if isinstance(result, dict) and not result.get("success", False):
                    return "FAILED"
                elif isinstance(result, dict):
                    # Check nested results (e.g., linting with multiple tools)
                    if any(not sub_result.get("success", False) 
                          for sub_result in result.values() 
                          if isinstance(sub_result, dict)):
                        return "FAILED"
        
        return "PASSED"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check unit test coverage
        if "unit_tests" in self.test_results:
            unit_result = self.test_results["unit_tests"]
            if unit_result.get("success") and "coverage" in unit_result.get("stdout", ""):
                # Parse coverage from output (simplified)
                if "85%" not in unit_result["stdout"]:
                    recommendations.append("Consider increasing unit test coverage to >85%")
        
        # Check performance test results
        if "performance_tests" in self.test_results:
            perf_result = self.test_results["performance_tests"]
            if not perf_result.get("success", True):
                recommendations.append("Performance tests failed - investigate system bottlenecks")
        
        # Check security issues
        if "security" in self.test_results:
            security_results = self.test_results["security"]
            if isinstance(security_results, dict):
                for tool, result in security_results.items():
                    if not result.get("success", True) and result.get("return_code", 0) > 0:
                        recommendations.append(f"Security issues found by {tool} - review and fix")
        
        # Check code quality
        if "linting" in self.test_results:
            linting_results = self.test_results["linting"]
            if isinstance(linting_results, dict):
                for tool, result in linting_results.items():
                    if not result.get("success", True):
                        recommendations.append(f"Code quality issues found by {tool} - run fixes")
        
        if not recommendations:
            recommendations.append("All tests passed! Consider adding more edge case tests.")
        
        return recommendations
    
    def _generate_html_report(self, summary: Dict):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-passed {{ color: green; font-weight: bold; }}
        .status-failed {{ color: red; font-weight: bold; }}
        .test-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 RAG System Test Report</h1>
        <p>Generated: {summary['test_run_summary']['timestamp']}</p>
        <p>Duration: {summary['test_run_summary']['total_duration']:.2f} seconds</p>
        <p class="status-{summary['overall_status'].lower()}">Overall Status: {summary['overall_status']}</p>
    </div>
    
    <div class="test-section">
        <h2>📊 Test Results Summary</h2>
        <table>
            <tr><th>Test Suite</th><th>Status</th><th>Duration</th></tr>
        """
        
        for test_type, result in summary['results'].items():
            if isinstance(result, dict) and 'success' in result:
                status = "PASSED" if result['success'] else "FAILED"
                duration = result.get('duration', 0)
                html_content += f"<tr><td>{test_type}</td><td class='status-{status.lower()}'>{status}</td><td>{duration:.2f}s</td></tr>"
        
        html_content += """
        </table>
    </div>
    
    <div class="recommendations">
        <h2>💡 Recommendations</h2>
        <ul>
        """
        
        for rec in summary['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
        </ul>
    </div>
    
</body>
</html>
        """
        
        with open("test-results/test-report.html", "w") as f:
            f.write(html_content)
    
    def _print_summary_to_console(self, summary: Dict):
        """Print test summary to console."""
        print("\n" + "="*60)
        print("🧪 TEST EXECUTION SUMMARY")
        print("="*60)
        
        print(f"⏱️  Total Duration: {summary['test_run_summary']['total_duration']:.2f} seconds")
        print(f"📅 Completed: {summary['test_run_summary']['timestamp']}")
        
        overall_status = summary['overall_status']
        if overall_status == "PASSED":
            print("✅ Overall Status: PASSED")
        else:
            print("❌ Overall Status: FAILED")
        
        print("\n📊 Test Suite Results:")
        for test_type, result in summary['results'].items():
            if isinstance(result, dict) and 'success' in result:
                status_icon = "✅" if result['success'] else "❌"
                duration = result.get('duration', 0)
                print(f"  {status_icon} {test_type}: {duration:.2f}s")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*60)
        print(f"📄 Full report available at: test-results/test-report.html")
        print("="*60)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Enterprise RAG Chatbot Test Runner")
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--evaluation", action="store_true", help="Run evaluation tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--security", action="store_true", help="Run security analysis only")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmarks only")
    
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation tests")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    
    parser.add_argument("--dataset-path", type=str, help="Path to evaluation dataset")
    parser.add_argument("--performance-duration", type=int, default=60, 
                       help="Performance test duration in seconds")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.all or not any([args.unit, args.integration, args.evaluation, 
                               args.performance, args.security, args.lint, args.benchmarks]):
            # Run all tests by default
            results = runner.run_all_tests(
                skip_performance=args.skip_performance,
                skip_evaluation=args.skip_evaluation,
                coverage=not args.no_coverage
            )
        else:
            # Run specific test suites
            results = {}
            
            if args.unit:
                results["unit_tests"] = runner.run_unit_tests(coverage=not args.no_coverage)
            
            if args.integration:
                results["integration_tests"] = runner.run_integration_tests()
            
            if args.evaluation:
                results["evaluation_tests"] = runner.run_evaluation_tests(args.dataset_path)
            
            if args.performance:
                results["performance_tests"] = runner.run_performance_tests(args.performance_duration)
            
            if args.security:
                results["security"] = runner.run_security_tests()
            
            if args.lint:
                results["linting"] = runner.run_linting()
            
            if args.benchmarks:
                results["benchmarks"] = runner.run_benchmarks()
        
        # Exit with appropriate code
        if any(not result.get("success", False) 
               for result in results.values() 
               if isinstance(result, dict) and "success" in result):
            sys.exit(1)
        else:
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()