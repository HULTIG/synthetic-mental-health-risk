#!/bin/bash
cd /home/claude/jbi
python3 sim/experiments.py coupling signal bench > results/exp1.log 2>&1
python3 sim/experiments.py xai ablation > results/exp2.log 2>&1
python3 sim/experiments.py size kernel > results/exp3.log 2>&1
echo ALLDONE >> results/exp3.log
