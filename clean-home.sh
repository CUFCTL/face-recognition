#!/bin/bash

#This script will serve as a cleaning tool. After running the test
#script, this will delete any files and directories that were created
#in the process, but are not meant to push to master.

rm -rf test_images train_images orl_faces
rm *.o

#remove other items? binaries?
