import os
import sys

# Ensure project root is on sys.path so `frontend.front` can be imported
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Importing this module runs the Streamlit app defined in `frontend/front.py`
import frontend.front  # noqa: F401

