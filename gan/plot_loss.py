import os, sys
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Check how to re-split
# ('dLossReal:', 0.0, 'dLossFake:', 2.4400328e-13)

filename = sys.argv[1]

loss_real = []
loss_fake = []

with open(filename, 'r') as f:
	for line in f:
		if 'dLossReal' in line and 'dLossFake' in line:
			#line = re.split('(\'|:\',|,|)\n', line)
			line = line.split(',')
			l_real = line[1]
			l_fake = line[3]
			l_fake = l_fake.split('\n')[0]
			l_fake = l_fake.split(')')[0]
			l_real = float(l_real)
			l_fake = float(l_fake)
			print l_real, l_fake
			loss_real.append(l_real)
			loss_fake.append(l_fake)

# plot x and y