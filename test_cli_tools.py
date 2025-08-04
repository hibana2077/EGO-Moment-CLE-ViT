#!/usr/bin/env python3
"""
Test script for CLI tools

This script tests all the CLI tools to ensure they work correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_test(description, command, expect_success=True):
    """Run a test command and report results"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        
        if expect_success:
            if result.returncode == 0:
                print("[PASS] Command succeeded")
                return True
            else:
                print(f"[FAIL] Command failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False
        else:
            # For tests where we expect failure
            if result.returncode != 0:
                print("[PASS] Command failed as expected")
                return True
            else:
                print("[FAIL] Command succeeded when it should have failed")
                return False
                
    except subprocess.TimeoutExpired:
        print("[FAIL] Command timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Exception occurred: {e}")
        return False

def main():
    """Run all CLI tool tests"""
    print("🧪 CLI Tools Test Suite")
    print("="*60)
    
    # Change to project root directory
    os.chdir(Path(__file__).parent)
    
    tests = []
    
    # Test 1: download_simple.py list function
    tests.append(("List datasets (simple)", "python download_simple.py --list"))
    
    # Test 2: download_simple.py info function
    tests.append(("Get dataset info (simple)", "python download_simple.py --info cotton80"))
    
    # Test 3: download_simple.py with invalid dataset (should fail)
    tests.append(("Invalid dataset (simple)", "python download_simple.py --info invalid_dataset", False))
    
    # Test 4: setup_and_run.py dependency check
    tests.append(("Check dependencies", "python setup_and_run.py --check-only"))
    
    # Test 5: quick_start.py check
    tests.append(("Quick start check", "python quick_start.py --check"))
    
    # Test 6: quick_start.py architecture
    tests.append(("Quick start architecture", "python quick_start.py --arch"))
    
    # Test 7: setup_and_run.py dataset availability check
    tests.append(("Dataset availability check", "python setup_and_run.py --dataset cotton80 --download-only"))
    
    # Test 8: Help messages
    tests.append(("download_simple.py help", "python download_simple.py --help"))
    tests.append(("setup_and_run.py help", "python setup_and_run.py --help"))
    
    # Run all tests
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests, 1):
        if len(test) == 3:
            description, command, expect_success = test
        else:
            description, command = test
            expect_success = True
            
        print(f"\n[{i}/{total}] Running test...")
        if run_test(description, command, expect_success):
            passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
