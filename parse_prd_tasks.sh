#!/bin/bash

# Navigate to the football betting system directory
cd /home/ryan/code/football_betting_system

# Check if Task Master binary exists
if [ ! -f "/home/ryan/code/task-master-windsurf/taskmaster-windsurf" ]; then
    echo "❌ Task Master binary not found at /home/ryan/code/task-master-windsurf/taskmaster-windsurf"
    exit 1
fi

# Create docs directory if it doesn't exist
mkdir -p ./docs

# Copy our PRD to the expected location
cp ./.taskmaster/docs/prd.txt ./docs/PRD.text

echo "🎯 Parsing PRD with Windsurf Task Master..."
echo "📍 PRD Location: ./docs/PRD.text"
echo "🔧 Using binary: /home/ryan/code/task-master-windsurf/taskmaster-windsurf"

# Parse PRD into tasks with research mode for better task generation
/home/ryan/code/task-master-windsurf/taskmaster-windsurf --prd --research --num-tasks 35 --sidebar --force

echo ""
echo "✅ Tasks generated! Check ./.taskmaster/tasks/tasks.json"
echo ""
echo "📋 Next steps:"
echo "   • List tasks: /home/ryan/code/task-master-windsurf/bin/task-master.js list"
echo "   • Next task: /home/ryan/code/task-master-windsurf/bin/task-master.js next"
echo "   • Set status: /home/ryan/code/task-master-windsurf/bin/task-master.js set-status --id=1 --status=done"
