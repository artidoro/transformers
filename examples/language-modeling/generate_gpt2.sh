#!/bin/bash

for gpu in {0..3}; do
    echo "The gpu number is $gpu"
    python generate_text.py $gpu &
done
    

