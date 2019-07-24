#!/usr/bin/python.

import numpy as np

def arrays2file(arrays, filename):
	s = "x\t" + "\t".join([ "y%d" %i for i in range(len(arrays)-1)]) + "\n"
	for i in range(arrays[0].shape[0]):
		s += "\t".join(map(str, [a[i] for a in arrays])) + "\n"
	f = open(filename, "w")
	f.write(s)
	f.close()
