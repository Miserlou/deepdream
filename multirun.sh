#!/bin/bash

for f in ./images/* ./images/**/* ; do
  python deepdreaming.py -s $f -i 50;
done;
