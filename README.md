<h1 align="center">Chatbot</h1>


![](https://3.bp.blogspot.com/-3Pbj_dvt0Vo/V-qe-Nl6P5I/AAAAAAAABQc/z0_6WtVWtvARtMk0i9_AtLeyyGyV6AI4wCLcB/s1600/nmt-model-fast.gif)


## Project Desription:

By learning a large number of sequence pairs, this model generates one from the other. More kindly explained, the I/O of Seq2Seq is below:
* Input: sentence of text data e.g. “How are you doing?”
* Output: sentence of text data as well e.g. “Not so bad.”

For training our seq2seq model, we will use Cornell Movie — Dialogs Corpus Dataset which contains over 220,579 conversational exchanges between 10,292 pairs of movie characters. And it involves 9,035 characters from 617 movies.
Then we will input these pairs of conversations into Encoder and Decoder.
The Layers can be broken down into 5 different parts:
* Input Layer (Encoder and Decoder)
* Embedding Layer (Encoder and Decoder)
* LSTM Layer (Encoder and Decoder)
* Decoder Output Layer

Thereby we designed and trained a Deep NLP model on a Seq2Seq Architecture to create a chatbot using the TensorFlow RNN(LSTM) model.

## Outputs: 

| Question				| Answer				|
| ------------- | ------------- |
| bill id like you to meet jack torrance | how much do you think |
| what is it | i dont know |
| when are you going to let the police know | you dont know what youre talking about |
| you can do it | yeah i think so |
| you would take a pain n the arse full stop | dont worry about it |
| nah i dont think so | why not |
| make sure he doesnt leave | you sure |
| do we need him | i dont know what to do |
| shall i leave | of course |
| jesus christ you scared the shit out of me | whats going on |
| im going to miss you | no youre not |
| you fucking bastard | come on |
| did i wake you | no you dont |
| that can wait till the weekend | are you kidding |
| maybe if i kiss him ill feel it | then you should have to |
| look fry company says were responsible for every one of those | dont be silly |
| you sent for me | yes yes i am |


## Dependencies:
1. Python: 3.5
2. Tensorflow: 1.0.0
3. Numpy: 1.14.3


