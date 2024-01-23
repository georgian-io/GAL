# Alignment

Large language models (LLMs) are often trained on large amounts of data that generally includes information from the internet. This training process means that LLMs may exhibit behaviors that they see in the data. Unfortunately, the internet as a whole is not the greatest example of "good" behavior. Thus, having a method to guide the LLM to generate outputs in a [helpful, honest and harmless manner](https://arxiv.org/abs/2204.05862) can be important. Alignment is this process of "aligning" the LLM to human standards. In general, human standards refers to a combination of helpfulness (assist users), honesty (do not make up things/use only truthful information) and harmlessness (do not be offensive or discriminatory). This is called the [HHH Framework](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf).

Alignment can be a challenging task. Even if we were to find a dataset of human behavior and fine tune a model on it, there's no guarantee that the model can achieve our criteria. For instance, let’s say we have two responses to a given prompt. Both might be technically true but one uses discriminatory language while the other doesn't. How can we help the model learn the reason one prompt is preferred over the other? This is where alignment methods such as Reinforcement Learning from Human Feedback (RLHF) come in.

# Reinforcement Learning from Human Feedback (RLHF)

RLHF is a method of fine-tuning LLMs that has grown into recent popularity with several Large Language Models (LLMs) including OpenAI's ChatGPT, Anthropic's Claude and DeepMind's Sparrow utilizing some variation of it. It consists of three steps that combine supervised training and reinforcement learning. 

![RLHF Overview diagram from OpenAI's ChatGPT blog post](images/ChatGPT_Diagram.svg "RLHF Overview")
RLHF Overview from OpenAI's [ChatGPT blogpost.](https://openai.com/blog/chatgpt)

## Step 1: Supervised Fine Tuning (SFT)

This step involves fine-tuning the LLM on data for some particular task. This task could include summarization or question-answering. 

## Step 2: Training a Reward Model (RM)

This step involves training a separate model to rank preferences. Specifically, given a prompt and two outputs, the model outputs a scalar score for each output. It uses a pairwise loss function that takes in the scores for each output along with human labels (the higher ranked output is given a label of 1 while the other gets a label of 0). 

The loss function is defined as: 

$loss(r_\theta) = -log (\sigma(r_\theta(x, y_i) - r_\theta(x, y_j)) $

where $r_\theta$ is the model, $x$ is the prompt and $y_i, y_j$ are the higher and lower ranked outputs respectively. 

Note: Practically, we may have several outputs for every given prompt but only sample two of these prompts at a time.

## Step 3: Reinforcement Learning from Human Feedback (RLHF/Step-3)

Note: This step is often just referred to as step 3 since the overall technique is called RLHF. 

The third step utilizes reinforcement learning to train the model from step 1 (sometimes called the policy or actor model) via reinforcement learning. Specifically, this process uses the Proximal Policy Optimization (PPO) algorithm. A rollout consists of sampling a random prompt and generating an output using the policy model. The reward is then calculated using the reward model. The policy model is then updated using this reward via the PPO algorithm.

## LLaMa 2 RLHF

LLaMa 2 introduced some [changes](https://arxiv.org/abs/2307.09288) to the RLHF paradigm. Specifically, the authors used multiple reward models each offering a reward for different criteria - namely safety and helpfulness. The final reward function is simply a linear combination of these reward models. 

In addition, the data is annotated differently than OpenAI's approach. Instead of ranking responses and then picking two of them at each training step, only two responses are annotated for each prompt. However, there is also an extra "margin" label that ranges from "significantly better" to "negligibly better". This margin label can then be used during the training step as a "margin loss". 

Finally, LLaMa 2 uses something known as rejection sampling. In rejection sampling, K outputs are sampled from the model and the output with the highest reward (from the reward model) is chosen. In step 3 of RLHF, they train 5 successive versions (with a new batch of preference data used for each model) of the RLHF model. The first 4 exclusively use rejection sampling and in the final model, they use a combination of rejection sampling and PPO. The authors state that rejection sampling acts as a breadth search due to multiple sampling while PPO acts as depth search. 

# Reinforcement Learning from AI Feedback (RLAIF)

RLAIF is similar to RLHF except that it replaces human feedback with AI feedback. Here AI feedback refers to a model, usually some kind of LLM, providing the feedback. The input to this feedback-providing LLM is structured such that there's a preamble detailing the task and instructions, few-shot (solved examples of similar tasks) examples, a sample to annotate and an ending string to prompt the LLM ("Preferred Answer = ").

Sometimes, the preamble can be more detailed and act as a "constitution". The constitution acts as a set of principles defined in natural language for the LLM to follow while providing feedback. We can thus describe the qualities of desirable outputs to the LLM in order to make it more helpful and harmless. In scenarios where all the answers offered to the reward model contain undesirable properties, a dataset cleaning process can be undertaken. In this process, the LLM generates revisions of a given answer while the AI Feedback model determines if there are undesirable qualities to the generated revision. When undesirable properties are identified, the AI Feedback model provides feedback to the LLM on why the revision was rejected.

# Other Alignment Methods

Alignment is a fairly recent topic. While this document covers some approaches, there may be others worth looking into. For instance, [Rank Response to align Human Feedback (RRHF)](https://github.com/GanjinZero/RRHF) and [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) both claim comparable performance to RLHF-based models with simpler training paradigms.

# Frequently Asked Questions

* How is the first step (SFT) different from just fine-tuning a model like BERT?
  
  It isn't. The first step is the same as fine-tuning older LLMs such as BERT. 

* Why train a reward model and perform reinforcement learning? Why not just use the human preferences to directly train the LLM?

  We train a reward model because it allows us to obtain human-life preferences for data the model has not seen before. This gives us a larger pool of data with which we can train the LLM. 

  For most tasks that use RLHF, there is usually no singular right or wrong answer. Since there are different ways of answering a given question, using supervised fine-tuning which emphasizes one answer above others does not help as much as reinforcement learning which rewards all good answers.

* Manually having annotators rank outputs is both expensive and time-consuming. What are my alternatives?

  In some cases, we might already have the rankings we need. For example, the [Stack Exchange Preferences Dataset](https://huggingface.co/datasets/lvwerra/stack-exchange-paired) consists of pairs of answers to Stack Exchange questions. Since the platform already uses a voting system, these existing votes can be used as the ranking instead. Similarly, other websites such as reddit also have a voting system and thus could be used in place of human annotators. There are also existing open source datasets such as [Anthropic's hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).

* Some of these models have billions of parameters. How do they train these models efficiently?

  Most of the LLMs in the news are enormous and are usually too big to fit on a single GPU. As a solution to this, large models generally use a combination of distributed training strategies alongside other techniques. Some common strategies include [parallelism](https://huggingface.co/docs/transformers/v4.17.0/en/parallelism), [Low Rank Adaptation (LoRA)](https://huggingface.co/docs/diffusers/training/lora) and [8-bit matrix multiplication](https://huggingface.co/blog/hf-bitsandbytes-integration).

* How do we select human annotators such that they represent a diversity of regional, cultutral and other backgrounds? This is important to prevent unwanted biases from seeping into the model.

  Appendix B of the [InstructGPT paper](https://arxiv.org/abs/2203.02155) goes into detail on how OpenAI approached this problem.

* Since SFT is similar to fine-tuning older LLMs, it is likely that data already exists. However, not a lot of datasets exist for training the reward model. If a company wanted to setup their own LLM on a custom dataset, how much labeled data would they require to train a reward model?

  [InstructGPT (section A.3)](https://arxiv.org/abs/2203.02155) uses around 50k prompts in total. Considering that each of these have anywhere from 4 to 9 responses, the number of training data points (tuples of prompt, winning response, losing response) is in the order of anywhere from a few 100k to almost 2M!

* How many prompts would we need for step 3?

  InstructGPT uses about 40,000. Based on our research, we believe you would need anything between 10k to 100k prompts.

* How do we validate the human preferences obtained for step 2? If we have 10 different outputs for a prompt, people are likely to rank them in different orders due to their own biases. Thus we can't use a single person's ranking. But how many people do we need for this? Do we use a majority ranking or an average? 

  [InstructGPT (section 5.3)](https://arxiv.org/abs/2203.02155) used around 40 labelers but there was rarely any overlap in the comparisons that they labeled. OpenAI acknowledges that this isn't ideal but does note that the labelers tended to agree with each other roughly 70% of the time. They also note that simply taking the average preference doesn't always work. For example when generating text that affects a minority group negatively, that group's preferences should be weighed more heavily.
  
* How big of a model do we need to get good results?

  From our experience, the specific model and dataset(s) used is more important in determining performance. Consult the [Chatbot Arena Leaderboard](https://lmsys.org/blog/2023-05-25-leaderboard/). We see that in some cases smaller models outperform larger ones (such as Vicuna-7B outperforming Alpaca-13B). This does imply that we could get away with training a 7B parameter model and get good performance. However, note that larger models do tend to do better as seen in GPT, Claude and PaLM.

* What hyper-parameters make the most impact on the results?

  The folks in charge of DeepSpeed have an [entire document](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/README.md) detailing their study on this topic.

* What is a good number for perplexity/accuracy/reward for each of the steps?

  We found that a perplexity of around 1.8 for step 1, a reward model accuracy of over 60% and an average reward of at least 5 tended to result in a model that worked relatively well in our tests. However, this is by no means a guarantee that your model will work well. According to the [DeepSpeed repository](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/README.md), although the perplexity stabilized within 1-2 epochs, they found that training it for longer (around 16 epochs) resulted in better generation quality.

* What is the smallest working model that we can train on readily available GPUs like a T4 used in Colab Free?

  Although we haven't been able to train an LLM on a T4 like in Google Colab, we found that we could train a 7B parameter model at a relatively low cost (~$17). You can find more details about this in the `rlhf_quickstart.md` documented located in this repository.

# Open Questions

This section consists of interesting questions that we don't yet have an answer to/are in the process of figuring out.


# Resources

* [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static): An example of the kind of data that the reward model training requires.
* [https://github.com/heartexlabs/RLHF/blob/master/tutorials/RLHF_with_Custom_Datasets.ipynb](RLHF With Custom Datasets): A quick guide to using RLHF with a custom dataset.
* [https://huyenchip.com/2023/05/02/rlhf.html](RLHF: Reinforcement Learning from Human Feedback): Blog post from Chip Huyen.
* [wandb rlhf tutorial](https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-trlX--VmlldzozMzAwODM2): A tutorial on training a summarization model via RLHF.
* [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat): DeepSpeed is a deep learning optimization library. It also offers a relatively easy way to get started with RLHF.
* [Hugging Face's StackLLaMa](https://huggingface.co/blog/stackllama): An instructive blog post on training LLaMA with RLHF.
* [Rank Response to align Human Feedback (RRHF)](https://github.com/GanjinZero/RRHF): This is an alternative to RLHF that is in active development. They claim comparable performance to RLHF-based models with a simpler training paradigm.
* [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267): Compares RLAIF vs RLHG empirically.
* [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073): Introduces the idea of constitutional AI
* [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives): A comprehensive blog post exploring RLHF in-depth.
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288): The LLaMa 2 paper.
* [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290): Another alternative to RLHF.