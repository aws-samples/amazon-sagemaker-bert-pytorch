## Fine tune a PyTorch BERT model and deploy it with Elastic Inference on Amazon SageMaker

### Background and Motivation

Text classification is a technique for putting text into different categories and has a wide range
of applications: email providers use text classification to detect to spam emails, marketing
agencies use it for sentiment analysis of customer reviews, and moderators of discussion forums use
it to detect inappropriate comments.

In the past, data scientists used methods such as [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf),
[word2vec](https://en.wikipedia.org/wiki/Word2vec), or [bag-of-words (BOW)](https://en.wikipedia.org/wiki/Bag-of-words_model)
to generate features for training classification models. While these techniques have been very
successful in many NLP tasks, they don't always capture the meanings of words accurately when they
appear in different contexts. Recently, we see increasing interest in using Bidirectional Encoder
Representations from Transformers (BERT) to achieve better results in text classification tasks,
due to its ability more accurately encode the meaning of words in different contexts.

Amazon SageMaker is a fully managed service that provides developers and data scientists with the
ability to build, train, and deploy machine learning (ML) models quickly. Amazon SageMaker removes
the heavy lifting from each step of the machine learning process to make it easier to develop
high-quality models. The SageMaker Python SDK provides open source APIs and containers that make it
easy to train and deploy models in Amazon SageMaker with several different machine learning and
deep learning frameworks. We use an Amazon SageMaker Notebook Instance for running the code.
For information on how to use Amazon SageMaker Notebook Instances,
see the [AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html).

Our customers often ask for quick fine-tuning and easy deployment of their NLP models. Furthermore,
customers prefer low inference latency and low model inference cost.
[Amazon Elastic Inference](https://aws.amazon.com/machine-learning/elastic-inference/) enables
attaching GPU-powered inference acceleration to endpoints, reducing the cost of deep learning
inference without sacrificing performance.

The notebook in this repository demonstrates how to use Amazon SageMaker to fine tune a PyTorch
BERT model and deploy it with Elastic Inference. We walk through our dataset, the training process,
and finally model deployment. This work is inspired by a post by
[Chris McCormick and Nick Ryan](https://mccormickml.com/2019/07/22/BERT-fine-tuning/).

### What is BERT?

First published in November 2018, BERT is a revolutionary model. First, one or more words in
sentences are intentionally masked. BERT takes in these masked sentences as input and trains itself
to predict the masked word. In addition, BERT uses a "next sentence prediction" task that pre-trains
text-pair representations. BERT is a substantial breakthrough and has helped researchers and data
engineers across industry to achieve state-of-art results in many Natural Language Processing (NLP)
tasks. BERT offers representation of each word conditioned on its context (rest of the sentence).
For more information about BERT, please refer to [1].

### BERT fine tuning

One of the biggest challenges data scientists face for NLP projects is lack of training data; they
often have only a few thousand pieces of human-labeled text data for their model training. However,
modern deep learning NLP tasks require a large amount of labeled data. One way to solve this problem
is to use transfer learning.

Transfer learning is a machine learning method where a pre-trained model, such as a pre-trained
ResNet model for image classification, is reused as the starting point for a different but related
problem. By reusing parameters from pre-trained models, one can save significant amounts of training
time and cost.

BERT was trained on BookCorpus and English Wikipedia data, which contain 800 million words and
2,500 million words, respectively [2]. Training BERT from scratch would be prohibitively expensive.
By taking advantage of transfer learning, one can quickly fine tune BERT for another use case with a
relatively small amount of training data to achieve state-of-the-art results for common NLP tasks,
such as text classification and question answering.

### Reference

[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,
https://arxiv.org/pdf/1810.04805.pdf

[2] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and
Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching
movies and reading books. In Proceedings of the IEEE international conference on computer vision,
pages 19–27.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
