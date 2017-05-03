# caption_generator: An image captioning project
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/anuragmishracse/caption_generator/blob/master/LICENSE)

To generate a caption for any image in natural language, English. The architecture for the model is inspired from [1] by Vinyals et al. The module is built using [keras](https://keras.io/), the deep learning library. 

This repository serves two purposes:
- present/ discuss my model and results I obtained
- provide a simple architecture for image captioning to the community

## Model 

The Image captioning model has been implemented using the Sequential API of keras. It consists of two parts:
- __An encoder CNN model__: A pre-trained CNN is used to encode an image to its features. In this implementation VGG16 model[d] is used as encoder and with its pretrained weights loaded. The last softmax layer of VGG16 is removed and the vector of dimention (4096,) is obtained from the second last layer. 

_To speed up my training, I pre-encoded each image to its feature set. This is done in the `prepare_dataset.py` file to form a resultant pickle file `encoded_images.p`_

- __A word embedding model__:

- __A decoder RNN model__:

<div align="center">
  <img src="vis/model.png"><br><br>
</div>

## Dataset and trained weights

## Generating captions

## Results

----------------------------------

## Requirements 

## Scripts 

## Usage

----------------------------------

## References 
[1] Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)

[2]	Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier. Collecting Image Annotations Using Amazon's Mechanical Turk. In Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and Language Data with Amazon's Mechanical Turk.


## Acknowledgements

[a] I am thankful to my project guide Prof. NK Bansode and a big shoutout to my project teammates. We have also developed an implementation of [1] in TensorFlow available at [image-caption-generator](https://github.com/neural-nuts/image-caption-generator) which had been trained and tested on MS COCO dataset.

[b] Special thanks to [Ashwanth Kumar](https://github.com/ashwanthkumar) for helping me with the resources and effort to train my models. 

[c] [Keras: Deep Learning library for Theano and TensorFlow](https://keras.io/): Thanks to [François Chollet](https://github.com/fchollet) for developing and maintaining such a wonderful library.

[d] [deep-learning-models](https://github.com/fchollet/deep-learning-models): Thanks to [François Chollet](https://github.com/fchollet) for providing pretrained VGG16 model and weights. 

