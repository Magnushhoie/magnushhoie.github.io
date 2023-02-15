---
type: blog
layout: post
comments: true
title: Attention RNNs in Lasagne
excerpt: RNNs with ways to attend to specific parts of the input sequence have attracted a lot of attention recently. In this blog post I'll study the benefits of adding attention to a RNN model and how it can be implemented using Lasagne.
date: 2015-08-21T11:00:00.000Z
mathjax: true
published: true
---

#!!!WORK IN PROGRESS!!!#

RNNs using a encoder-decoder structure have recently shown great performance in a number of different tasks including Neural Machine Transalation and Image Caption generations. In this blog post I will show how such a model can be coded in [Lasagne](https://github.com/Lasagne/Lasagne). Furhter I'll demonstrate some of the limitations of of the simple ncoder-Decoder structure and how they can be fixed using specific attention to various parts of the input sequence$.$

## Encoder-Decoder
In the encoder-decoder structure one RNN (blue) encodes the input and a second RNN (red) calculates the target values. One essential step is to let the encoder and decoder communicate. The simplest approach is to use the last hidden state of the encoder as input to the the decoder. This means that the Encoder must compress all the knowledge of the input sequence into a fixed length vector (size: *number of neurons*) which is then used to produce the targets by the decoder. For long sequences this means that the Encoder needs to carry the information of the input sequence across many timesteps which requires many hidden units to store the information.


<div class="imgcap">
<img src="/assets/attentionRNNs/enc-dec.png">
<div class="thecap">
Encoder in Blue, Decoder in Red
</div>
</div>

### The Data
Since RNN models can be very slow to train on large datasets we will generate some simpler simulated data for this exercise. The task for the RNN is simply to translate a string of letters spelling the numbers between 0-9 into the corresponding numbers i.e

```"one two five" --> "125#" (we use # as a special stop of sequence character)```

To input the strings into the RNN model we translate the characters into a vector integers using a simple translation table (i.e. 'h'->16, 'o'-> 17 etc). See below for a data example 

```
  TEXT INPUTS:		nine eight
  TEXT TARGETS:		98#
  ENCODED INPUTS:	[18 15 18 12 11 12 15 13 16 22]
  MASK INPUTS:		[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
  ENCODED TARGETS:	[ 9  8 10]
  MASK TARGETS:		[ 1.  1.  1.]
```
See [data_generator.py](/assets/attentionRNNs/data_generator.py) for further details.

### Implementing an Encoder-Decoder RNN in Lasagne
Lasagne have implementations of LSTM and GRU units used in RNNs. Both layers assume that the input from the layer below have the shape **(Batchsize, seqlen, numfeatures)**. In this blog post I'll primarlily use the GRU unit since it only stores a single hidden value per neuron (LSTMs stores two) and is also faster to calculate compared LSTMs

We will first create an Encoder which encodes the input sequence.  The last hidden state of the Encoder is then used as input to the decoder model which then uses this information(simply a fixed length vector of numbers) to produce the targets. There is (at least) two different ways to input $h^{enc}_T$ into the decoder

1. Repeatly use $h^{enc}_T$ as input to the Decoder at each decode time step
2. Intialize the decoder using $h^{enc}_T$ and run the decoder without any inputs

Here I'll follow the first approach using a [repeatLayer](/assets/attentionRNNs/repeatLayer.py) in Lasagne. Below is the model specification using Lasagne

```python
BATCH_SIZE = 100
NUM_UNITS_ENC = 10
NUM_UNITS_DEC = 10
MAX_DIGITS = 20 
MIN_DIGITS = MAX_DIGITS #currently only support for same length outputs - we'll leave it for an exercise to add support for varying length targets
NUM_INPUTS = 27
NUM_OUTPUTS = 11 #(0-9 + '#')


#symbolic theano variables. Note that we are using imatrix for X since it goes into the embedding layer
x_sym = T.imatrix()
y_sym = T.imatrix()
xmask_sym = T.matrix()

##### ENCODER START #####
l_in = lasagne.layers.InputLayer((None, None))
l_emb = lasagne.layers.EmbeddingLayer(l_in, NUM_INPUTS, NUM_INPUTS, 
                                      W=np.eye(NUM_INPUTS,dtype='float32'),
                                      name='Embedding')
#Here we'll remove the trainable parameters from the embeding layer to constrain 
#it to a simple "one-hot-encoding". You can experiment with removing this line
l_emb.params[l_emb.W].remove('trainable') 
l_enc = lasagne.layers.GRULayer(l_emb, num_units=NUM_UNITS_ENC, name='GRUEncoder')

# slice last index of dimension 1
l_last_hid = lasagne.layers.SliceLayer(l_enc, indices=-1, axis=1)
##### END OF ENCODER######


##### START OF DECODER######
#note that the decoder have its own input layer, we'll use that to plug in the output 
#from the encoder later
l_in_dec = lasagne.layers.InputLayer(l_last_hid.output_shape) 
l_in_rep = RepeatLayer(l_in_dec, n=MAX_DIGITS+1) #we add one to allow space for the end of sequence character
l_dec = lasagne.layers.GRULayer(l_in_rep, num_units=NUM_UNITS_DEC, name='GRUDecoder')

# We need to do some reshape voodo to connect a softmax layer to the decoder.
# See http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html#examples 
# In short this line changes the shape from 
# (batch_size, decode_len, num_dec_units) -> (batch_size*decodelen,num_dec_units). 
# We need to do this since the softmax is applied to the last dimension and we want to 
# softmax the output at each position individually
l_reshape = lasagne.layers.ReshapeLayer(l_dec, (-1, [2]))
l_softmax = lasagne.layers.DenseLayer(l_reshape, num_units=NUM_OUTPUTS, 
                                      nonlinearity=lasagne.nonlinearities.softmax,
                                      name='SoftmaxOutput')
# reshape back to 3d format (batch_size, decode_len, num_dec_units). Here we tied the batch size to the shape of the symbolic variable for X allowing 
#us to use different batch sizes in the model.
l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTPUTS))
###END OF DECODER######
```


### Results Encoder-Decoder without attention

### Attention Encoder-Decoder

### Results Attention Encoder-Decoder

### Some results from protein data




