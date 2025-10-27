#!/bin/bash
# Stop any llama.cpp server running on port 8765

PORT=8765

PID=$(lsof -t -i:$PORT 2>/dev/null)

if [ -z "$PID" ]; then
    echo "No server running on port $PORT"
    exit 0
fi

echo "Stopping server (PID: $PID)..."
kill $PID

sleep 1

# Check if it's still running
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Server still running, force killing..."
    kill -9 $PID
fi

echo "✓ Server stopped"
