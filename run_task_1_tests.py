#!/usr/bin/env python3
"""
Execute Task 1 tests 3 times to ensure reliability
"""

import sys
import traceback
from test_task_1 import run_all_task_1_tests

def main():
    """Run Task 1 tests 3 times."""
    print("🚀 TURBO MODE: Running Task 1 tests 3 times")
    print("=" * 70)
    
    for i in range(1, 4):
        print(f"\n🧪 TEST RUN {i}/3")
        print("-" * 50)
        
        try:
            success = run_all_task_1_tests()
            if success:
                print(f"✅ Test Run {i}: PASSED")
            else:
                print(f"❌ Test Run {i}: FAILED")
                return False
        except Exception as e:
            print(f"❌ Test Run {i}: ERROR - {e}")
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 70)
    print("🎉 ALL 3 TEST RUNS COMPLETED SUCCESSFULLY")
    print("✅ Task 1: Setup Project Repository and CI/CD - COMPLETE")
    print("📋 Ready to proceed to Task 2: Python Environment and Dependency Management")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
