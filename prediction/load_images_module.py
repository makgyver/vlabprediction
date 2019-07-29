# -*- coding: utf-8 -*-

import sys
import glob
import numpy as np
#import matplotlib.pyplot as plt

def import_img(filename, N, verbose=True):
	print("Importing %s" %filename)
	#OPEN FILE
	f = open(filename, "r")
	content = f.readlines()
	f.close()

	#PREPROCESS
	img, row, raw = [], [], []
	for e, line in enumerate(content):
		if row and not line.startswith("    "):
			img.append(row)
			row = []
		#print(line)
		line = line.strip(" ,\n")
		raw = map(float, line.split(", "))
		row += [i if i <= 1. else 0. for i in raw]
		if verbose and not (e+1) % 10000: 
			sys.stdout.write("\rProcessed %d out of %d lines" %(e+1, len(content)))
	img.append(row)

	#SPLITTING
	rpi = int(np.array(img).shape[0]/N)
	#imgs = [np.array(img[i*rpi:(i+1)*rpi][::-1]) for i in range(N)]
	imgs = [np.array(img[i*rpi:(i+1)*rpi]) for i in range(N)]

	#PLOT
	'''
	if verbose:
		for i in range(10):
			print("Image %d with size %s" %(i+1,imgs[i].shape))
			plt.figure("Image %d" %(i+1))
			plt.imshow(imgs[i], cmap='gray')
			plt.show()
	'''
	return imgs


#def import_all(N, verbose):
#	files = sorted(glob.glob("permz-*"))
#	return [import_img(f, N, verbose)[0] for f in files] #REVIEW THIS


#INPUT
#filename = sys.argv[1]
#N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

#images = import_img(filename, N)
