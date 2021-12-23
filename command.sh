#!/bin/sh

# 5-way 1-shot
python3 main.py -N 5 -K 1 -B 4 --tb_port 5856 &

# 5-way 5-shot
python3 main.py -N 5 -K 5 -B 2 --tb_port 5857 &
