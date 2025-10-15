#!/usr/bin/bash

myarray=(50 100 200)

for i in "${myarray[@]}"; do
  python ./datanalysis.py $i
  python ./profitanalysis.py $i
done

for i in "${myarray[@]}"; do
  kitten icat $i.png
done
