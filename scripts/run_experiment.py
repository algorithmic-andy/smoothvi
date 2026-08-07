"""
Run the complete Experimental inference study.
"""

from __future__ import annotations

import json
from pathlib import Path

from .run_phase1 import main as phase1_main
from .run_phase2 import main as phase2_main


def main():

    phase1_main()

    phase2_main()


if __name__ == "__main__":

    main()