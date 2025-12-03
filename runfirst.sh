#!/usr/bin/bash

myarray=(50 100 200 250)

for i in "${myarray[@]}"; do
  python dataanalysis.py $i
  python profitanalysis.py $i
done

for i in "${myarray[@]}"; do
  wezterm imgcat $i.png
done
