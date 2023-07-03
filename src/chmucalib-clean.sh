#!/bin/bash
# arg $1 is location and arg $2 is model (perez or prague)
timeout 60m python3 main.py -mI perez-azimuth-independent -mJ "$2" -n 1000 -W 720 -H 540 -f0 500 -df date-time -I ../data/Ismall/"$1" -J ../data/Jsmall/"$1" -m ../data/masks/"$1".jpg -min 10 -max 240 -webcams ../data/webcams.json || echo "$1 TIMED OUT"
