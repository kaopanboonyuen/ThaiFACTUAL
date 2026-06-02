#!/bin/bash
# =============================================================================
# ThaiFACTUAL: Full Reproduction Pipeline
# =============================================================================
# Paper : Debiasing LLMs in Thai Political Stance Detection via
#          Counterfactual Calibration
#          EMNLP 2025 – Widening NLP (WiNLP) Workshop
# Author: Teerapong Panboonyuen (teerapong.pa@chula.ac.th)
# =============================================================================

set -e
echo "============================================================"
echo "  ThaiFACTUAL — Full Reproduction Pipeline"
echo "  EMNLP 2025 WiNLP Workshop"
echo "  Author: Teerapong Panboonyuen (Kao Panboonyuen)"
echo "============================================================"

# Create output dirs
mkdir -p results figures data

# Step 1: Build dataset (Appendix A)
echo ""
echo "[Step 1/5] Building Thai Political Stance Dataset (§A)..."
python data/dataset_builder.py

# Step 2: Inter-annotator agreement (§D.2)
echo ""
echo "[Step 2/5] Computing Fleiss' κ (paper target: 0.84) (§D.2)..."
python data/annotator_agreement.py

# Step 3: Build counterfactuals (Appendix B)
echo ""
echo "[Step 3/5] Generating Counterfactual Pairs (§B)..."
python models/counterfactual_builder.py

# Step 4: Run ThaiFACTUAL calibrator (§4)
echo ""
echo "[Step 4/5] Running ThaiFACTUAL Calibration (§4)..."
python models/thaifactual_calibrator.py

# Step 5: Full evaluation — reproduce Table 1
echo ""
echo "[Step 5/5] Evaluating All Models — Reproducing Table 1 (§3.3)..."
python evaluation/metrics.py

echo ""
echo "============================================================"
echo "  ✓ All done! Results saved to: results/ and figures/"
echo "  ✓ Table 1 bar chart: figures/table1_comparison.png"
echo "============================================================"
