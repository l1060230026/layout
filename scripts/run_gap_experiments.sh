#!/bin/bash
# Script to run gap analysis experiments
# This addresses reviewer concerns about gaps under complex constraints

echo "=========================================="
echo "Gap Analysis Experiment for House Diffusion"
echo "=========================================="
echo ""
echo "This experiment will:"
echo "1. Test occupancy ratios under varying constraint levels (0-5 fixed rooms)"
echo "2. Validate multi-candidate selection mechanism (N=1,3,5,10,20)"
echo "3. Generate statistical reports and visualizations"
echo "4. Create a response letter for the reviewer"
echo ""
echo "Estimated time: 30-60 minutes depending on your GPU"
echo ""

# Default parameters
NUM_SAMPLES=20
NUM_CANDIDATES=10
MODEL_PATH="ckpts/rplan_7/model500000.pt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --candidates)
            NUM_CANDIDATES="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --quick)
            NUM_SAMPLES=5
            echo "Quick mode: Using only 5 samples for faster testing"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--samples N] [--candidates N] [--model PATH] [--quick]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Number of test samples: $NUM_SAMPLES"
echo "  Number of candidates: $NUM_CANDIDATES"
echo "  Model path: $MODEL_PATH"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please specify the correct model path with --model option"
    exit 1
fi

# Create output directory
mkdir -p outputs/gap_analysis

# Run the experiment
echo "Starting experiments..."
echo ""

python scripts/test_gap_analysis.py \
    --model_path "$MODEL_PATH" \
    --num_test_samples "$NUM_SAMPLES" \
    --num_candidates "$NUM_CANDIDATES" \
    --dataset_length 1000 \
    --output_dir outputs/gap_analysis

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Experiments completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to outputs/gap_analysis/:"
    echo "  📊 results.json - Detailed numerical results"
    echo "  📈 constraint_analysis.png - Occupancy vs constraint level"
    echo "  📈 candidate_effectiveness.png - Multi-candidate effectiveness"
    echo "  📈 improvement_distribution.png - Distribution of improvements"
    echo "  📝 response_letter.txt - Response to reviewer"
    echo ""
    echo "You can now:"
    echo "  1. Review the plots in outputs/gap_analysis/"
    echo "  2. Read the response letter: cat outputs/gap_analysis/response_letter.txt"
    echo "  3. Include the figures in your rebuttal/revised paper"
    echo ""
else
    echo ""
    echo "Error: Experiments failed. Please check the error messages above."
    exit 1
fi
