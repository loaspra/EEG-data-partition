#!/usr/bin/env python3
"""
Simple test script to run the validation suite.
"""

import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

try:
    import config
    print("✓ Config imported successfully")
    print("Results directory:", config.RESULTS_DIR)
except ImportError as e:
    print(f"✗ Config import failed: {e}")
    sys.exit(1)

try:
    from utils import load_subject_data, train_min2net_translator
    print("✓ Utils imported successfully")
except ImportError as e:
    print(f"✗ Utils import failed: {e}")
    sys.exit(1)

try:
    from mlp_classifier import MLPClassifier
    print("✓ MLP Classifier imported successfully")
except ImportError as e:
    print(f"✗ MLP Classifier import failed: {e}")
    sys.exit(1)

try:
    import validation_suite
    print("✓ Validation suite imported successfully")
except ImportError as e:
    print(f"✗ Validation suite import failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("Running validation suite...")
print("="*50)

try:
    validator = validation_suite.run_comprehensive_validation()
    print("✓ Validation completed successfully!")
except Exception as e:
    print(f"✗ Validation failed: {e}")
    import traceback
    traceback.print_exc() 