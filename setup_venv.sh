#!/usr/bin/env bash
# Creates a .venv in this directory and installs everything needed to run
# the GasExperiment python files (Gas_Classification.py, FeatureSelection.py,
# preprocess.py, Get_Gas_Data.py, Plot_Gas_Experiment.py).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python3.10 not found, falling back to python3" >&2
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "Done. Activate with: source .venv/bin/activate"
echo ""
echo "Note: preprocess.py and Plot_Gas_Experiment.py import PhytoNode/utils"
echo "from the sibling DataProcessing repo, not from this requirements.txt."
echo "Run them with that repo on PYTHONPATH, e.g.:"
echo "  PYTHONPATH=../DataProcessing:../DataProcessing/PhytoNode .venv/bin/python preprocess.py"
