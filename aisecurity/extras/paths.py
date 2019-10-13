
"""

"paths.py"

Common paths used throughout the repository.

"""

import os
import aisecurity

try:
  HOME = os.path.abspath(aisecurity.__file__).replace("/__init__.py", "/")
except FileNotFoundError:
  raise FileNotFoundError("aisecurity repository not found. Please move to ~/PycharmProjects or ~/Desktop")