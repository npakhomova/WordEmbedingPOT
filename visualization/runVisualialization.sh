#!/usr/bin/env bash
# please see details from http://mylearning9.com/?p=4

python prepareForTsNE.py -e ../data/dresses_long_description_vector.txt -s ../data/dresses_long_description_dictionary.txt -l ../data/visualization/label_w2v.txt -x ../data/visualization/embed_w2v.txt

python tsnePlot.py -l ../data/visualization/label_w2v.txt -v ../data/visualization/embed_w2v.txt -d 200 -p 50
