import sys
sys.path.append('..')
import numpy as np
from PIL import Image
import cv2
import caffe
from sklearn.decomposition import IncrementalPCA
image_name_list_file = sys.argv[1]
image_name_list = open(image_name_list_file)
caffe.set_mode_gpu()
caffe.set_device(0) 
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
image_number = int(sys.argv[2])
feature_all=[]
net = caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', caffe.TEST)
mean = np.load('imagenet_mean.npy')
mean = mean.transpose((1,2,0))
print mean.shape
mean = cv2.resize(mean,(227,227))
for i in range(0,image_number/10):
	img_batch=[]
	for j in range(0,10):	
		image_name = image_name_list.readline()[:-1]
		im = Image.open(image_name)
		
		in_ = np.array(im, dtype=np.float32)
		in_ = in_[:,:,::-1]#change image channel to BGR
		in_ = cv2.resize(in_,(227,227))
		in_ -= mean
		in_ = in_.transpose((2,0,1))
		img_batch.append(in_.copy())
		# load net
	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(10, *img_batch[0].shape)
	net.blobs['data'].data[...] = img_batch
	# run net and take argmax for prediction
	net.forward()
	conv3 = net.blobs['conv3'].data
	print conv3.shape
	conv3 = conv3.reshape(10,64896)
	feature_all.append(conv3.copy())
	print i
feature_all = np.array(feature_all)
feature_all = feature_all.reshape(image_number,64896)
ipca = IncrementalPCA(n_components=316)
ipca.fit(feature_all)
new_y=ipca.transform(feature_all)
np.savetxt('map_316_0501.txt',new_y)
np.savetxt('mean_0501.txt',ipca.mean_)
np.savetxt('comonents_0501.txt',ipca.components_)
