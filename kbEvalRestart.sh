#!/bin/bash

SECONDS_THRESHOLD=3600

TARGET_PIDS=$(ps -eo pid,etimes,cmd   | awk -v T="$SECONDS_THRESHOLD" '/[p]ython.*kbEvalServer.py/ && $2 > T' | grep -v while | awk '{print $1}')

if [ -z "$TARGET_PIDS" ]; then
    echo "No target processes found"
    exit 1
fi

echo "Killing target processes: $TARGET_PIDS"
kill -15 $TARGET_PIDS

sleep 3
kill -9 $TARGET_PIDS