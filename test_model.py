import caption_generator
import numpy as np
from keras.preprocessing import sequence

cg = caption_generator.CaptionGenerator()

def get_caption_from_index(caption):
	return " ".join([cg.index_word[index] for index in caption])

def generate_caption(model, image, beam_size):
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
				print "a-->"+str(len(temp_captions))
		captions = temp_captions
		captions.sort(key = lambda l:l[1])
		captions = captions[-beam_size:]
		print "b->>"+str(len(captions))
		print len(captions[0][0])


	#if(cg.index_word[next_word]=='<end>'):
	#	break
	return get_caption_from_index(caption)


def test_model(weight, img_path, beam_size = 3):
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)

	image = cg.load_image(img_path)
	caption = generate_caption(model, image, beam_size)
	return caption


if __name__ == '__main__':
	print test_model('weights-improvement-01.hdf5', 'Flicker8k_Dataset/2513260012_03d33305cf.jpg')