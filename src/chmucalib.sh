#!/bin/bash
# $1 is location and $2 is model (perez or prague)
timeout 30m python3 main.py -mJ "$2" -n 1000 -W 720 -H 540 -f0 500 -df date-time -I ../data/I/"$1" -J ../data/J/"$1" -m ../data/sky-masks/"$1".jpg || echo "$1 TIMED OUT"
