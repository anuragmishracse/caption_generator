import cPickle as pickle
import caption_generator
import numpy as np
from keras.preprocessing import sequence
import nltk

cg = caption_generator.CaptionGenerator()

def process_caption(caption):
	caption_split = caption.split()
	processed_caption = caption_split[1:]
	try:
		end_index = processed_caption.index('<end>')
		processed_caption = processed_caption[:end_index]
	except:
		pass
	return " ".join([word for word in processed_caption])

def get_best_caption(captions):
    captions.sort(key = lambda l:l[1])
    best_caption = captions[-1][0]
    return " ".join([cg.index_word[index] for index in best_caption])

def get_all_captions(captions):
    final_captions = []
    captions.sort(key = lambda l:l[1])
    for caption in captions:
        text_caption = " ".join([cg.index_word[index] for index in caption[0]])
        final_captions.append([text_caption, caption[1]])
    return final_captions

def generate_captions(model, image, beam_size):
	start = [cg.word_index['<start>']]
	captions = [[start,0.0]]
	while(len(captions[0][0]) < cg.max_cap_len):
		temp_captions = []
		for caption in captions:
			partial_caption = sequence.pad_sequences([caption[0]], maxlen=cg.max_cap_len, padding='post')
			next_words_pred = model.predict([np.asarray([image]), np.asarray(partial_caption)])[0]
			next_words = np.argsort(next_words_pred)[-beam_size:]
			for word in next_words:
				new_partial_caption, new_partial_caption_prob = caption[0][:], caption[1]
				new_partial_caption.append(word)
				new_partial_caption_prob+=next_words_pred[word]
				temp_captions.append([new_partial_caption,new_partial_caption_prob])
		captions = temp_captions
		captions.sort(key = lambda l:l[1])
		captions = captions[-beam_size:]

	return captions

def test_model(weight, img_name, beam_size = 3):
	encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)

	image = encoded_images[img_name]
	captions = generate_captions(model, image, beam_size)
	return process_caption(get_best_caption(captions))
	#return [process_caption(caption[0]) for caption in get_all_captions(captions)] 

def bleu_score(hypotheses, references):
	return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)

def test_model_on_images(weight, img_dir, beam_size = 3):
	imgs = []
	captions = {}
	with open(img_dir, 'rb') as f_images:
		imgs = f_images.read().strip().split('\n')
	encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)

	f_pred_caption = open('predicted_captions.txt', 'wb')

	for count, img_name in enumerate(imgs):
		print "Predicting for image: "+str(count)
		image = encoded_images[img_name]
		image_captions = generate_captions(model, image, beam_size)
		best_caption = process_caption(get_best_caption(image_captions))
		captions[img_name] = best_caption
		print img_name+" : "+str(best_caption)
		f_pred_caption.write(img_name+"\t"+str(best_caption))
		f_pred_caption.flush()
	f_pred_caption.close()

	f_captions = open('Flickr8k_text/Flickr8k.token.txt', 'rb')
	captions_text = f_captions.read().strip().split('\n')
	image_captions_pair = {}
	for row in captions_text:
		row = row.split("\t")
		row[0] = row[0][:len(row[0])-2]
		try:
			image_captions_pair[row[0]].append(row[1])
		except:
			image_captions_pair[row[0]] = [row[1]]
	f_captions.close()
	
	hypotheses=[]
	references = []
	for img_name in imgs:
		hypothesis = captions[img_name]
		reference = image_captions_pair[img_name]
		hypotheses.append(hypothesis)
		references.append(reference)

	return bleu_score(hypotheses, references)

if __name__ == '__main__':
	weight = 'weights-improvement-48.hdf5'
	test_image = '3155451946_c0862c70cb.jpg'
	test_img_dir = 'Flickr8k_text/Flickr_8k.testImages.txt'
	#print test_model(weight, test_image)
	print test_model_on_images(weight, test_img_dir, beam_size=3)
