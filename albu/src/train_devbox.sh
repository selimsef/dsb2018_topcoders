#!/usr/bin/env bash
tmux set-option remain-on-exit on
tmux split-window -t 0 -p 50 "./train01.sh $1"

sleep 20
./train23.sh $1
