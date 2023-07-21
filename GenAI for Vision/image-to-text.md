# Text-to-Image

Image-to-text, as the name suggests, is getting a text output given an image input. There are several different tasks that utilize this such as Visual Question Answering (VQA), image captioning or even Optical Character Recognition (OCR). 


## Table of Contents

1. [VisualBERT](#1-visualbert)
2. [Vision Transformer](#2-vision-transformer)
3. [Vision Encoder Decoder Model](#3-vision-encoder-decoder-model)
4. [CLIP](#4-clip)
5. [BLIP](#5-blip)
6. [Getting Started](#6-getting-started)
7. [Resources](#7-resources)

## 1. [VisualBERT](https://arxiv.org/pdf/1908.03557.pdf)

![VisualBERT Architecture](images/visualbert_architecture.png)
Source: [VisualBERT paper](https://arxiv.org/pdf/1908.03557.pdf)

VisualBERT is largely based on the original [BERT](https://jalammar.github.io/illustrated-bert/) architecture. This is further modified to include a set of visual embeddings. A CNN acts as an object detector and identifies entities in the image. The corresponding embedding is then obtained from a pre-trained CNN such as a ResNet and then sent into the BERT model.

More specifically, each input embedding is the sum of three separate embeddings - the position, segment, and token embedding. The token embedding is the respective text or image embedding for that particular token (where a detected object in the image is considered a token). The segment embedding indicates if the embedding is textual or visual. The position embedding is the standard positional embedding for text. For images, it is the sum of the positional embeddings of text that is aligned with the image region (if provided). The rest of the training loop proceeds as usual.

The model is usually trained on image+caption data such as [COCO](https://cocodataset.org/). First, the model is pre-trained in a task-agnostic manner. This is followed by task-specific pre-training and finally task-specific fine-tuning.

You can get the VisualBERT model as well as several fine-tuned checkpoints for specific tasks through [HuggingFace](https://huggingface.co/docs/transformers/model_doc/visual_bert).

## 2. [Vision Transformer](https://arxiv.org/abs/2010.11929)

![Vision Transformer Architecture](images/vit_architecture.png)
Source: [Vision Transformer paper](https://arxiv.org/abs/2010.11929)

The Vision Transformer (ViT) is similar to VisualBERT in that it uses a transformer-based architecture for image data. More specifically, it uses a transformer encoder. The image is split into fixed-size patches which are then flattened by passing them through a linear layer. Position embeddings are added to each patch projection and the sequence is fed as an input to a standard transformer encoder. Inspired by BERT, a learnable `[class]` embedding is prepended to the sequence which is then fed into a classification head. This model is pre-trained in a supervised fashion with image labels and then fine-tuned on an image classification dataset. 

At a high level, we can think of the ViT model as on that applies a transformer to image classification. Since breaking the entire image down into pixels and using attention with a large sequence length (a 64x64 image would need an attention matrix of 4096x4096) would not be feasible, the image is broken down into smaller patches. Thus the attention matrix is smaller and the model is faster (presumably at the cost of some performance). 

In terms of performance, the ViT outperforms CNNs only if there is a lot of data (~10 million). Otherwise the CNN generally wins out. ViT also has a comparitively lower training time. While the vision transformer isn't used on its own often, it is important to understand since many future models build on it. 

There are several spin-offs of the ViT model such as BEiT (BERT Pre-Training of Image Transformers) which essentially performs self-supervised pre-training instead of supervised pre-training.

## 3. [Vision Encoder Decoder Model](https://huggingface.co/docs/transformers/main/model_doc/vision-encoder-decoder)

The Vision Encoder Decoder Model is a generic framework to create an image-to-text model using any pretrained transformer-based vision model as an encoder and any pretrained language model as the decoder. It is based on Microsoft's [TrOCR](https://arxiv.org/abs/2109.10282) model. Transformer based vision models include ViT and BEiT while the decoder can be something like GPT or BERT. 

This model is interesting because it builds on two different models - one to process image data and the other to process text data. It then simply combines the functionalities of these two models to perform a particular task. For instance, the [vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) model uses a ViT encoder and GPT-2 decoder and is trained for image captioning. While not particularly a state of the art model anymore, it does give a good idea of how we can build new models from existing ones.

## 4. [CLIP](https://arxiv.org/abs/2103.00020)

CLIP (Contrastive Language-Image Pre-training) contains both an image and text encoder. It takes as input an image and a caption, encodes them and compares the embeddings using a (cosine) similarity metric. The model is then trained on this similarity task (=1 when the image and caption are related and =0 otherwise). After training for a long while, our encoder reaches a point where the image embedding for a picture of a dog and the text embedding for "a picture of a dog" are very similar. To better train the model, negative samples are also used (sample an image and unrelated caption or vice-versa). We can then give it a phrase and acquire an embedding in the image space that corresponds to that phrase (or vice-versa). This process is known as contrastive learning. The encoders may vary based on the specific model but in general the text encoder is some kind of transformer model and the image encoder is usually a ResNet or a Vision Transformer.

## 5. [BLIP](https://arxiv.org/abs/2201.12086)

![BLIP Architecture](images/blip_architecture.png)
Source: [BLIP paper](https://arxiv.org/abs/2201.12086)

BLIP (Bootstrapping Language-Image Pre-training) was created by Salesforce as a general purpose vision-language model for both understanding and generation tasks including captioning, question-answering and reasoning. BLIP is a multimodal mixture of encoders and decoders that integrates three functionalities:

1. Unimodal Encoder: This attempts to encode the text and visual data separately through BERT and a Vision Transformer respectively. These two models are trained through Image-Text Contrastive (ITC) loss which aims to align the feature space of the two models. This is done by pushing similar image-text pairs to have similar representations. 

2. Image-Grounded Text Encoder: This inserts a cross-attention layer between the self-attention and feed forward layers of each transformer block in the text encoder. A task-specific [Encode] token is appended to the text, and the output embedding of [Encode] is used as the multimodal representation of the image-text pair. This block uses Image-Text Matching (ITM) loss to distinguish between positive and negative image-text pairs. ITM is a binary classification problem where the model just predicts if an image-text pair is matched or unmatched.

3. Image-Grounded Text Decoder: This replaces the bi-directional self-attention layers in the previous block with causal self-attention layers. A special [Decode] token is used to signal the beginning of a sequence. A standard Language Modeling (LM) loss is used here to generate text conditioned on an image. 

Note that the architecture above is color-coded to denote shared parameters. For instance, between the image-grounded text encoder and decoder, the parameters of the feedforward and cross attention layers are shared. The authors claim that this is because the differences between the encoding and decoding tasks are best captured by the self-attention layers. Thus the encoder uses bi-directional self-attention to build representations for the current input while the decoder uses causal self-attention to predict the next tokens. 

All three objectives (ITC, ITM, LM) are optimized during the pre-training process. Thus each image-text pair requires one forward pass of the vision transformer and three forward passes through the text transformer. 

[This](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/) blog from the Salesforce teams takes a deeper dive into this model. The BLIP model is a fairly robust model that works well for most vision-language tasks and thus is a good starting point for any work in this area. The model can be used through [HuggingFace](https://huggingface.co/Salesforce/blip-image-captioning-large).

## 6. Getting Started

We have two example notebooks for you to get started with! The first is `im2text_finetuning.ipynb` which walks you through the process of fine-tuning an image-to-text model (BLIP). The second is `im2text_applications.ipynb` which is a quick demo for you to see how image captioning and VQA works.

## 7. Resources

* [A Dive into Vision-Language Models](https://huggingface.co/blog/vision_language_pretraining): An informative blog from HuggingFace.

* [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/): A summary of the BLIP model from the Salesforce team.