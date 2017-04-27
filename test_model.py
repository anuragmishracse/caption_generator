import caption_generator
import numpy as np

cg = caption_generator.CaptionGenerator()

def get_caption_from_index(caption):
	return " ".join([cg.index_word(index) for index in caption])

def generate_caption(model, image):
	caption = [cg.word_index['<start>']]
	while(len(caption) < cg.max_cap_length):
		partial_caption = sequence.pad_sequences(caption, maxlen=self.max_cap_len, padding='post')
		next_word = np.argmax(model.predict([image, partial_caption]))
		caption.append(next_word)
		if(cg.index_word(next_word)=='<end>'):
			break
	return get_caption_from_index(caption)


def test_model(weight, img_path):
	model = cg.create_model(ret_model = True)
	model.load_weights(weight)

	image = cg.load_image(img_path)
	caption = generate_caption(model, image)
	return caption


if __name__ == '__main__':
	print test_model('', '')