#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as Math
import pylab
import argparse
import tsne

# important!!! : to change size of output picture - uncomment this code and play with multiplicator

# Get current size
fig_size = pylab.rcParams["figure.figsize"]



# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
multiplicator =10

# Set figure width to 12 and height to 9
fig_size[0] = fig_size[0]*multiplicator
fig_size[1] = fig_size[1]*multiplicator
pylab.rcParams["figure.figsize"] = fig_size

print ("New Figsize",  pylab.rcParams["figure.figsize"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store', dest='labelfile', required=True,
                        help='Path of label file')
    parser.add_argument('-v', action='store', dest='vectorFile', required=True,
                        help='Embedding vector')
    parser.add_argument('-d', action='store', type=int, dest='demension',
                        required=True, help='Demension of vector')
    parser.add_argument('-p', action='store', type=int, dest='perplexity',
                        default=20, help='Perplexity, usually between 20 to 50')
    r = parser.parse_args()

    X = Math.loadtxt(r.vectorFile)

    with open(r.labelfile, 'r') as f:
        labels = f.read().upper().splitlines()
        print('Reading labels...')
        Y = tsne.tsne(X, 2, r.demension, r.perplexity)
        fig, ax = pylab.subplots()
        ax.scatter(Y[:, 0], Y[:, 1], 20)

        for i, txt in enumerate(labels):
            ax.annotate(txt, (Y[:, 0][i], Y[:, 1][i]))
        pylab.savefig('../data/visualization/myModelSky.png', bbox_inches='tight')
        pylab.show()

