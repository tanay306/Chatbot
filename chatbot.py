# -*- coding: utf-8 -*-
"""
Created on Tue May 25 23:07:21 2021

@author: Tanay Gandhi
"""

# BUILDING A CHATBOT WITH DEEP NLP


# STEP 1: IMPORTING THE LIBRARIES

import numpy as np
import tensorflow as tf
import re
import time


# STEP 2: DATA PREPROCESSING

# IMPORTING THE DATASET

lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# CREATING A DICTIONARY THAT MAPS EACH LINES AND ITS ID

id2line = {}

for line in lines :
  # Splitting the line by the code word `+++$+++`
  _line = line.split(' +++$+++ ') 
  if len(_line) == 5 : 
      id2line[_line[0]] = _line[4]
      
# CREATING A LIST OF ALL OF THE CONVERSATIONS   
      
conversations_ids = []

for conversation in conversations[:-1] : 
  _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
  conversations_ids.append(_conversation.split(','))

# GET THE QUESTIONS AND THE ANSWERS SEPARATELY
  
questions = []
answers = []

for conversation in conversations_ids :
  for i in range (len(conversation) - 1) : 
    questions.append(id2line[conversation[i]])
    answers.append(id2line[conversation[i+1]])
    
# CLEANING ALL THE TEXTS FUNCTION
    
def clean_text(text):
    text = text.lower() # to lower case every letter in the text
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# CLEANING THE QUESTIONS
    
clean_questions = []

for question in questions : 
  clean_questions.append(clean_text(question))
  
# CLEANING THE ANSWERS
  
clean_answers = []

for answer in answers : 
  clean_answers.append(clean_text(answer))

# REMOVING THE NON FREQUENT WORDS OCCURING BY CREATING A CORPUS
# CREATING A DICTIONARY THAT MAPS EACH WORD TO ITS NUMBER OF OCCURANCES
  
word2count = {}

for question in clean_questions : 
  for word in question.split() :
    if word not in word2count : 
      word2count[word] = 1
    else :
      word2count[word] += 1

for answer in clean_answers : 
  for word in answer.split() :
    if word not in word2count : 
      word2count[word] = 1
    else :
      word2count[word] += 1
      
# CREATING TWO DICTIONARIES THAT MAP THE QUESTIONS WORDS AND ANSWERS WORDS TO A UNIQUE INTEGERS
# TOKENISATION AND FILTERING THE NON FREQUENT WORDS

threshold = 20
questionsWords2int = {}
word_number = 0

for word, count in word2count.items() :
  if count >= threshold : 
    questionsWords2int[word] = word_number
    word_number += 1

answersWords2int = {}   
word_number = 0

for word, count in word2count.items() :
  if count >= threshold : 
    answersWords2int[word] = word_number
    word_number += 1      
      
# ADDING THE LAST TOKENS TO THESE DICTIONARIES

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionsWords2int[token] = len(questionsWords2int) + 1
for token in tokens:
    answersWords2int[token] = len(answersWords2int) + 1      
      
# CREATING THE INVERSE DICTIONARY OF THE ANSWERSWORDS2INT DICTIONARY
# This is to inverse map the integers in the answerWords2int dictionary to individual words.

answersInt2Words = {w_i : w for w, w_i in answersWords2int.items()}   

# ADDING THE <EOS> TOKEN TO THE END OF EVERY ANSWERS IN THE LIST

for i in range (len(clean_answers)) :
  clean_answers[i] += ' <EOS>'  
  
# TRANSLATING ALL THE QUESTIONS AND ANSWERS TO INTEGERS AND REPLACING ALL THE WORDS WHICH WERE FILTERED OUT BY <OUT>
  
questions_to_int = []

for question in clean_questions :
  ints = []
  for word in question.split() :
    if word not in questionsWords2int :
      ints.append(questionsWords2int['<OUT>'])
    else :
      ints.append(questionsWords2int[word])
  questions_to_int.append(ints) 
  
answers_to_int = []

for answer in clean_answers :
  ints = []
  for word in answer.split() :
    if word not in answersWords2int :
      ints.append(answersWords2int['<OUT>'])
    else :
      ints.append(answersWords2int[word])
  answers_to_int.append(ints) 
      
# SORTING THE QUESTIONS AND ANSWERS BY THE LENGTH OF THE QUESTIONS
  
sorted_clean_questions = []
sorted_clean_answers = []

# Min length = 1, Max length = 24
for length in range (1, 25) : 
  for i in enumerate(questions_to_int) :
    if len(i[1]) == length :
      sorted_clean_questions.append(questions_to_int[i[0]])
      sorted_clean_answers.append(answers_to_int[i[0]])
      

# STEP 3: BUILDING THE NLP SEQUENCE TO SEQUENCE MODEL

# CREATING PLACEHOLDERS FOR THE INPUTS AND THE TARGETS    
      
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

# PREPROCESSING THE TARGETS INTO BATCHES OF SOME SIZE

def preprocess_targets(targets, word2int, batch_size):
    # First word should be a starting token
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    # Stride and slide each target
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

# CREATING THE ENCODING RNN LAYER { STACKED LSTM }
      
# RNN Inputs: Inputs, targets, lr etc.
# RNN Size: Number of input tensors
# RNN Layers
# Keep_Prob: To control the dropout rate
# Sequence_Length: length of questions

def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    # Input and ouput size of the forward and backward cell must match
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state

# DECODING THE OBSERVATIONS ON THE TRAINING SET
  
# Encoder state: output from encoder_rnn_layer
# Decoder cell: cell in the rnn decoder
# Decoder embedded unit: unit on which embedding is applied on
# Sequence_length
# Decoding_Scope: ds that wraps tf variables
# Output_function: fn used to return decoder o/p
# Keep Prob
# Batch_size
# (Embedding: Mapping from discrete objects like words to real number vectors)
  
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    # Training for dynamic rnn decoder
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# DECODING THE OBSERVATIONS IN THE TEST/VALIDATION STATE
  
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    # Testing for dynamic rnn decoder
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions

# CREATING THE DECODING RNN LAYER
  
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        # Initialize Weights
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        # Get the biases
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

# BUILDING THE SEQUENCE TO SEQUENCE MODEL
        
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    # Putting together the encoder and the decoder
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                          answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionsWords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


# STEP 4: TRAINING THE SEQUENCE TO SEQUENCE MODEL
  
# SETTING THE HYPERPARAMETERS (Based upon several experimentations)
  
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
# To reduce the learning_rate to learn in depth
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# DEFINING A SESSION IN TENSORFLOW

tf.reset_default_graph()
session = tf.InteractiveSession()
      
# LOADING THE MODEL INPUTS OF THE SEQUENCE TO SEQUENCE MODEL     

inputs, targets, lr, keep_prob = model_inputs()

# SETTING THE SEQUENCE LENGTH TO MAXIMUM LENGTH

sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# GETTING THE SHAPE OF THE INPUT TENSOR

input_shape = tf.shape(inputs)

# GETTING THE TRAINING AND THE TEST PREDICTIONS

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answersWords2int),
                                                       len(questionsWords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionsWords2int)
      