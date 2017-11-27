import json, sys, os
import commands
import cv2
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Reading ui_names.json and find the different image sizes in the rico dataset.
###############################################################################
def find_different_image_sizes_in_rico_dataset(filepath):
	filepath = sys.argv[1]
	names = json.loads(open(filepath).read())
	sizes = ['1080x1920', '540x960']
 
	for key, value in names.iteritems():
		print key, len(value)
		value = sorted(value)
		for i in range(len(value)):
			tmp_value = value[i].split('.')[0]
			command = "convert " + "rico-dataset/combined/" + tmp_value + ".jpg -print '%wx%h\n' /dev/null"
			output = commands.getoutput(command)
			print "image", value[i], output
			if output not in sizes:
				sizes.append(output)
	print "output", sizes
	return sizes


######################################################
# Parse json file and find all the bounding box sizes
######################################################
bounds_list = []

def findbounds(inp):
	if type(inp) == dict:
		visible_to_user = False
		for key, value in inp.iteritems():
			if key == 'visible-to-user' and value == True:
				visible_to_user = True
		# for key, value in inp.iteritems():
		# 	if key == 'content-desc' and visible_to_user == True:
		# 		print key, value
		for key, value in inp.iteritems():
			if key == 'text' and value == True:
				print key, value
		for key, value in inp.iteritems():
			if key == 'bounds' and visible_to_user == True:
				bounds_list.append(value)
				visible_to_user = False
			elif type(value) == dict:
				findbounds(inp[key])
			elif key == "children":
				findbounds(value)
	elif type(inp) == list:
		for i in range(len(inp)):
			if type(inp[i]) == dict:
				findbounds(inp[i])
			elif type(inp[i]) == list:
				findbounds(inp[i])

#############################################
# Utility functions
#############################################

def recover_image(layout_image, input_txt):
	img = cv2.imread(layout_image)
	matf = np.fromfile(input_txt, dtype=int, count=-1, sep='\n')

	matr = matf[0:len(matf)/2]
	matb = matf[len(matf)/2:len(matf)]

	matr = matr.reshape((100,56))
	matb = matb.reshape((100,56))

	plt.subplot(131)
	plt.imshow(img)
	plt.subplot(132)
	plt.imshow(matr, cmap = 'gray')
	plt.subplot(133)
	plt.imshow(matb, cmap = 'gray')
	plt.title('recovered input')
	plt.show()

	matr = np.array(matr).astype('uint8')
	matb = np.array(matb).astype('uint8')

	h,w = img.shape[:2]
	print w,h
	matr = cv2.resize(matr, (w, h), interpolation = cv2.INTER_CUBIC)
	matb = cv2.resize(matb, (w, h), interpolation = cv2.INTER_CUBIC)

	print img.shape, matr.shape, matb.shape
	plt.subplot(131)
	plt.imshow(img)
	plt.subplot(132)
	plt.imshow(matr, cmap = 'gray')
	plt.subplot(133)
	plt.imshow(matb, cmap = 'gray')
	plt.title('resized recovered input')
	plt.show()


filenames = []
matrix = []

def generate_input(layout_image, name):
	img = cv2.imread(layout_image)
	matr = np.zeros(img.shape[:2])
	matb = np.zeros(img.shape[:2])
	maskr = np.all([img[:,:,0] > 240, img[:,:,1] < 20, img[:,:,2] < 20], axis=0)
	maskb = np.all([img[:,:,0] < 20, img[:,:,1] < 20, img[:,:,2] > 240], axis=0)

	matr[maskr] = 1
	matb[maskb] = 1

	matr = np.array(matr).astype('uint8')
	matb = np.array(matb).astype('uint8')

	# plt.subplot(131)
	# plt.imshow(img)
	# plt.subplot(132)
	# plt.imshow(matr, cmap = 'gray')
	# plt.subplot(133)
	# plt.imshow(matb, cmap = 'gray')
	# plt.title('generated input')
	# plt.show()

	matr = cv2.resize(matr,(56, 100), interpolation = cv2.INTER_CUBIC)
	matb = cv2.resize(matb,(56, 100), interpolation = cv2.INTER_CUBIC)

	matr = matr.reshape(-1)
	matb = matb.reshape(-1)
	matf = np.hstack((matr, matb))
	global matrix
	if matrix == []:
		matrix = matf
	else:
		matrix = np.vstack((matrix, matf))
	print matrix.shape
	#######################
	# Write in a file
	#######################
	#np.savetxt(name, matf, delimiter=',',fmt='%d')

'''
# count = 1
path = sys.argv[1]
files = os.listdir(path)
for f in sorted(files):
	if f.endswith('_layout.jpg'):
		name = f.split('.')[0]
		name = name + '_input.txt'
		filenames.append(f)
		print f
		generate_input(os.path.join(path,f), os.path.join(path,name))
		#recover_image(os.path.join(path,f), os.path.join(path,name))
		# if count == 10:
		# 	break
		# count = count + 1


np.savetxt("out_matrix.csv", matrix, delimiter=',',fmt='%d')
np.savetxt("out_names.csv", filenames, delimiter=',',fmt='%s')


# matrix = np.loadtxt("out_matrix.csv", dtype=int, delimiter=',')
# print matrix.shape
# filenames = np.loadtxt("out_names.csv", dtype=str, delimiter=',')
# print filenames.shape
'''