#!/bin/bash
# Run the ASL-Tutor API server

echo "Starting ASL-Tutor API Server..."
echo "================================"
echo ""

# Check if models exist
if [ ! -f "models/asl_cnn.pt" ] && [ ! -f "models/asl_cnn_best.pt" ]; then
    echo "Warning: No trained model found in models/"
    echo "Please train the model first using:"
    echo "  python -m src.train"
    echo "Or run the training notebook:"
    echo "  jupyter notebook notebooks/01_train_model.ipynb"
    echo ""
fi

# Start the server
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
