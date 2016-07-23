#!/usr/bin/python
# This script will take an input directory and crop all the faces in the
# subdirectories. Ideal use would be for orl_faces

import sys
import os
import subprocess
from time import sleep

if len(sys.argv) != 3:
    print '\n  ***USAGE: python crop.py [directory/to/be/cropped] [output/directory]***\n'
    sys.exit(1)

contents = os.listdir(sys.argv[1])

subprocess.call(["mkdir", sys.argv[2]])

print contents

print 'Cleaning...\n'
subprocess.call(["make", "clean"])

print 'Compiling...\n'
subprocess.call(["make"])

print 'Running ./detect on ' + sys.argv[1] + '\n'

i = 0.0

for subdir in contents:
    # print status of program
    sys.stdout.write('\r')
    sys.stdout.write("[%-*s] %d/%d %3.1f%%" % (len(contents), '='*int(i), i+1, len(contents), ((i+1.0)/len(contents))*100.0))
    sys.stdout.flush();

    # get proper paths for i/o, configure args
    path_in = sys.argv[1] + "/" + subdir
    path_out = sys.argv[2] + "/" + subdir

    args = ["./detect", path_in, path_out]

    # run detect on each subdirectory
    subprocess.call(args)

    i += 1

print '\n\nCropping complete, images in ' + sys.argv[2] + '\n'
