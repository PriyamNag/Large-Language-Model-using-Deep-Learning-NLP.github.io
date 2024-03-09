# Large-Language-Model-using-Deep-Learning-NLP.github.io
Final Year LLM model Project from scratch on Deep Learning: NLP using Pytorch

NLP is a field of linguistics and machine learning focused on understanding everything related to human language. The aim of NLP tasks is not only to understand single words individually, but to be able to understand the context of those words.

The following is a list of common NLP tasks, with some examples of each:

Classifying whole sentences: Getting the sentiment of a review, detecting if an email is spam, determining if a sentence is grammatically correct or whether two sentences are logically related or not
Classifying each word in a sentence: Identifying the grammatical components of a sentence (noun, verb, adjective), or the named entities (person, location, organization)
Generating text content: Completing a prompt with auto-generated text, filling in the blanks in a text with masked words
Extracting an answer from a text: Given a question and a context, extracting the answer to the question based on the information provided in the context
Generating a new sentence from an input text: Translating a text into another language, summarizing a text.

Multi-Layer Perceptron(MLP) is the simplest type of artificial neural network. It is a combination of multiple perceptron models. Perceptrons are inspired by the human brain and try to simulate its functionality to solve problems. In MLP, these perceptrons are highly interconnected and parallel in nature. This parallelization is helpful in faster computation.

Transformer models are used to solve all kinds of NLP tasks, like the ones mentioned in the previous section. Many MNCs and organizations using Hugging Face and Transformer models, who also contribute back to the community by sharing their models.

The Transformer architecture was introduced in June 2017. The focus of the original research was on translation tasks. This was followed by the introduction of several influential models, including:

June 2018: GPT, the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results

October 2018: BERT, another large pretrained model, this one designed to produce better summaries of sentences (more on this in the next chapter!)

February 2019: GPT-2, an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns

October 2019: DistilBERT, a distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT’s performance

October 2019: BART and T5, two large pretrained models using the same architecture as the original Transformer model (the first to do so)

May 2020, GPT-3, an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning)

This list is far from comprehensive, and is just meant to highlight a few of the different kinds of Transformer models. Broadly, they can be grouped into three categories:

GPT-like (also called auto-regressive Transformer models)
BERT-like (also called auto-encoding Transformer models)
BART/T5-like (also called sequence-to-sequence Transformer models)
We will dive into these families in more depth later on.

Transformers are language models
All the Transformer models mentioned above (GPT, BERT, BART, T5, etc.) have been trained as language models. This means they have been trained on large amounts of raw text in a self-supervised fashion. Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model. That means that humans are not needed to label the data!

This type of model develops a statistical understanding of the language it has been trained on, but it’s not very useful for specific practical tasks. Because of this, the general pretrained model then goes through a process called transfer learning. During this process, the model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task.

An example of a task is predicting the next word in a sentence having read the n previous words. This is called causal language modeling because the output depends on the past and present inputs, but not the future ones.

The model is primarily composed of two blocks:

Encoder : The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
Decoder : The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

A key feature of Transformer models is that they are built with special layers called attention layers.
