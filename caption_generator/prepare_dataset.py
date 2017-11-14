import cPickle as pickle
from keras.preprocessing import image
from vgg16 import VGG16
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input	

counter = 0

def load_image(path):
    img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.asarray(x)

def load_encoding_model():
	model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
	return model

def get_encoding(model, img):
	global counter
	counter += 1
	image = load_image('Flicker8k_Dataset/'+str(img))
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[1])
	print "Encoding image: "+str(counter)
	print pred.shape
	return pred

def prepare_dataset(no_imgs = -1):
	f_train_images = open('Flickr8k_text/Flickr_8k.trainImages.txt','rb')
	train_imgs = f_train_images.read().strip().split('\n') if no_imgs == -1 else f_train_images.read().strip().split('\n')[:no_imgs]
	f_train_images.close()

	f_test_images = open('Flickr8k_text/Flickr_8k.testImages.txt','rb')
	test_imgs = f_test_images.read().strip().split('\n') if no_imgs == -1 else f_test_images.read().strip().split('\n')[:no_imgs]
	f_test_images.close()

	f_train_dataset = open('Flickr8k_text/flickr_8k_train_dataset.txt','wb')
	f_train_dataset.write("image_id\tcaptions\n")

	f_test_dataset = open('Flickr8k_text/flickr_8k_test_dataset.txt','wb')
	f_test_dataset.write("image_id\tcaptions\n")

	f_captions = open('Flickr8k_text/Flickr8k.token.txt', 'rb')
	captions = f_captions.read().strip().split('\n')
	data = {}
	for row in captions:
		row = row.split("\t")
		row[0] = row[0][:len(row[0])-2]
		try:
			data[row[0]].append(row[1])
		except:
			data[row[0]] = [row[1]]
	f_captions.close()

	encoded_images = {}
	encoding_model = load_encoding_model()

	c_train = 0
	for img in train_imgs:
		encoded_images[img] = get_encoding(encoding_model, img)
		for capt in data[img]:
			caption = "<start> "+capt+" <end>"
			f_train_dataset.write(img+"\t"+caption+"\n")
			f_train_dataset.flush()
			c_train += 1
	f_train_dataset.close()

	c_test = 0
	for img in test_imgs:
		encoded_images[img] = get_encoding(encoding_model, img)
		for capt in data[img]:
			caption = "<start> "+capt+" <end>"
			f_test_dataset.write(img+"\t"+caption+"\n")
			f_test_dataset.flush()
			c_test += 1
	f_test_dataset.close()
	with open( "encoded_images.p", "wb" ) as pickle_f:
		pickle.dump( encoded_images, pickle_f )  
	return [c_train, c_test]

if __name__ == '__main__':
	c_train, c_test = prepare_dataset()
	print "Training samples = "+str(c_train)
	print "Test samples = "+str(c_test)
