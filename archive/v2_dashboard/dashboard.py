"""
Root launcher for the RF Environment Simulator Dashboard.
Runs the comprehensive dashboard located in the `dashboard/` directory.
"""
from __future__ import annotations

import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import and execute the app module
import dashboard.app
