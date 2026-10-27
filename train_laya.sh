#!/bin/bash
# Train Laya TSP Decision Model
# Modes:
#   ./train_laya.sh          -> Fast Mode (Frozen Encoder, ~6 mins on Apple Silicon)
#   ./train_laya.sh --epoch2 -> Train Epoch 2 from Epoch 1 Checkpoint (~4-5 hours)
#   ./train_laya.sh --full   -> Full Mode (Unfrozen Encoder from scratch, 4 epochs)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="fast"
if [ "$1" == "--epoch2" ] || [ "$1" == "--resume" ]; then
    MODE="epoch2"
elif [ "$1" == "--full" ] || [ "$1" == "--unfreeze" ]; then
    MODE="full"
fi

if [ "$MODE" == "epoch2" ]; then
    echo "=========================================================="
    echo " Starting Laya TSP Decision Model Training: EPOCH 2"
    echo " Base Model: model/laya_tsp_model/checkpoint_epoch1"
    echo " Configuration: Micro-batch 4 | Grad Accum 16 (Effective: 64)"
    echo " Learning Rates: Encoder 2e-6 | Head 5e-5"
    echo " Device: Apple Silicon GPU (MPS)"
    echo " Expected Time: ~9 hours"
    echo "=========================================================="

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 model/.venv/bin/python -m laya.train_cli \
        --data data/laya_train.jsonl \
        --eval data/laya_valid.jsonl \
        --base model/laya_tsp_model/checkpoint_epoch1 \
        --out model/laya_tsp_model \
        --epochs 1 \
        --micro-batch 4 \
        --grad-accum 16 \
        --encoder-lr 2e-6 \
        --head-lr 5e-5 \
        --loss rlcd \
        --label-smoothing 0.05 \
        --shuffle-options \
        --device mps
elif [ "$MODE" == "full" ]; then
    echo "=========================================================="
    echo " Starting Laya TSP Decision Model Training: FULL MODE"
    echo " Configuration: Unfrozen ModernBERT Encoder (Deep Fine-Tuning)"
    echo " Device: Apple Silicon GPU (MPS)"
    echo " Epochs: 4 | Micro-batch: 4 | Grad Accum: 16 (Effective: 64)"
    echo " Learning Rates: Encoder 5e-6 | Head 1e-4"
    echo " Loss: RLCD (Calibrated Policy Gradient) | Label Smoothing: 0.05"
    echo " Dataset: 46,342 train / 5,222 validation steps (Nodes: 4 to 500)"
    echo "=========================================================="

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 model/.venv/bin/python -m laya.train_cli \
        --data data/laya_train.jsonl \
        --eval data/laya_valid.jsonl \
        --out model/laya_tsp_model \
        --epochs 4 \
        --micro-batch 4 \
        --grad-accum 16 \
        --encoder-lr 5e-6 \
        --head-lr 1e-4 \
        --loss rlcd \
        --label-smoothing 0.05 \
        --shuffle-options \
        --device mps
else
    echo "=========================================================="
    echo " Starting Laya TSP Decision Model Training: FAST MODE"
    echo " Configuration: Frozen Encoder (10x Speedup, ~6 mins on Apple Silicon)"
    echo " Device: Apple Silicon GPU (MPS)"
    echo " Epochs: 3 | Micro-batch: 16 | Grad Accum: 4 (Effective: 64)"
    echo " Loss: RLCD (Calibrated Policy Gradient) | Label Smoothing: 0.05"
    echo " Dataset: 46,342 train / 5,222 validation steps (Nodes: 4 to 500)"
    echo "=========================================================="

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 model/.venv/bin/python -m laya.train_cli \
        --data data/laya_train.jsonl \
        --eval data/laya_valid.jsonl \
        --out model/laya_tsp_model \
        --epochs 3 \
        --micro-batch 16 \
        --grad-accum 4 \
        --freeze-encoder \
        --loss rlcd \
        --label-smoothing 0.05 \
        --shuffle-options \
        --device mps
fi

echo ""
echo "=========================================================="
echo " Training Complete! Checkpoint saved to: model/laya_tsp_model"
echo " You can now evaluate the model with: python3 solve_tsp_laya.py"
echo "=========================================================="
