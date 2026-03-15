import sys
import os

# Add src/ to sys.path so all test imports resolve regardless of invocation method.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
