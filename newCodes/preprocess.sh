#!/bin/bash

nohup python preprocess.py 20160801 > 20160801.out &
nohup python preprocess.py 20160901 > 20160901.out &
nohup python preprocess.py 20161001 > 20161001.out &
nohup python preprocess.py 20170801 > 20170801.out &
nohup python preprocess.py 20170901 > 20170901.out &
nohup python preprocess.py 20171001 > 20171001.out &
