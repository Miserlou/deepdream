#!/bin/bash

for f in ./images/* ./images/**/* ; do
  python deepdreamering.py -s $f -i 50;
done;
