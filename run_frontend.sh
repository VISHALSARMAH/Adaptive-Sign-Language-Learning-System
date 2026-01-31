#!/bin/bash
# Serve the ASL-Tutor frontend

echo "Starting ASL-Tutor Frontend Server..."
echo "======================================"
echo ""
echo "Frontend will be available at: http://localhost:3000"
echo ""
echo "Make sure the API server is running at http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd frontend
python -m http.server 3000
