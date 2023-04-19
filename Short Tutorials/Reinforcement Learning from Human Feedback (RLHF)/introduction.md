# Reinforcement Learning from Human Feedback (RLHF)

RLHF is one of the hottest topics of 2023 with several Large Language Models (LLMs) such as OpenAI's ChatGPT and DeepMind's Sparrow. It consists of three steps that combine supervised training and reinforcement learning. 

![RLHF Overview diagram from OpenAI's ChatGPT blogpost](images/ChatGPT_Diagram.svg "RLHF Overview")
RLHF Overview from OpenAI's [ChatGPT blogpost.](https://openai.com/blog/chatgpt)

## Step 1: Supervised Fine Tuning (SFT)

This step involves finetuning your LLM on data for some particular task. This task could be something like summarization or question-answering. 

## Step 2: Training a Reward Model (RM)

This step involves training a separate model to rank preferences. Specifically, given a prompt and two outputs, the model outputs a scalar score for each output. It uses a pairwise loss function that takes in the scores for each output along with human labels (the higher ranked output is given a label of 1 while the other gets a label of 0). 

The loss function is defined as: 

$loss(r_\theta) = -log (\sigma(r_\theta(x, y_i) - r_\theta(x, y_j)) $

where $r_\theta$ is the model, $x$ is the prompt and $y_i, y_j$ are the higher and lower ranked outputs respectively. 

Note: In reality, we may have several outputs for every given prompt but only sample two of these prompts at a time.

## Step 3: Reinforcement Learning from Human Feedback (RLHF/Step-3)

Note: This step is often just referred to as step 3 since the overall technique is called RLHF. 

The third step utilizes reinforcement learning to train the model from step 1 (sometimes called the policy or actor model) via reinforcement learning. Specifically, this uses the Proximal Policy Optimization (PPO) algorithm. A rollout consists of sampling a random prompt and generating an output using the policy model. The reward is then calculated using the reward model. The policy model is then updated using this reward via the PPO algorithm.

# Frequently Asked Questions

* How is the first step (SFT) different from just finetuning a model like BERT?
  
  It isn't! The first step is the same as finetuning older LLMs. 

* Why do we train a reward model and perform reinforcement learning? Why not just use the human preferences to directly train the LLM?

   There are a number of reasons for this. First is that of data. In a finetuning setting, we would be restricted to only samples that human have annotated. Using a reward model on the other hand allows us to obtain human-like preferences for unseen data as well. This means that we have a significantly larger pool of data to work with.

   The second reason is that we don't want to just train the model to give us any output. We want it to give us good outputs. For most of the tasks that use RLHF, there is no one right answer and there is no one wrong answer. Since there are so many different ways of answering a given question, using supervised finetuning which emphasizes one answer above all the others does not help as much as reinforcement learning which rewards all good answers.

* Manually having annotators rank outputs is both expensive and time-consuming. What are my alternatives?

  In some cases, we might already have the rankings we need. For example, the [Stack Exchange Preferences Dataset](https://huggingface.co/datasets/lvwerra/stack-exchange-paired) consists of pairs of answers to Stack Exchange questions. Since the platform already uses a voting system, these existing votes can be used as the ranking instead. There are also existing open source datasets such as [Anthropic's hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).

* Some of these models have billions of parameters. How do they train these models efficiently?

  Most of the LLMs in the news are enormous and are usually too big to fit on a single GPU. As a solution to this, large models generally use a combination of distributed training strategies alongside other techniques. Some common strategies include [parallelism](https://huggingface.co/docs/transformers/v4.17.0/en/parallelism), [Low Rank Adaptation (LoRA)](https://huggingface.co/docs/diffusers/training/lora) and [8-bit matrix multiplication](https://huggingface.co/blog/hf-bitsandbytes-integration).

* How do we select human annotators such that they represent a diversity of regional, cultutral and other backgrounds? This is important to prevent unwanted biases from seeping into the model.

  Appendix B of the [InstructGPT paper](https://arxiv.org/abs/2203.02155) goes into detail on how OpenAI approached this problem.

* Since SFT is similar to finetuning older LLMs, it is likely that data already exists. However, not a lot of datasets exist for training the reward model. If, say a company, wanted to setup their own LLM on a custom dataset, how much labeled data would they require to train a reward model?

  [InstructGPT (section A.3)](https://arxiv.org/abs/2203.02155) uses around 30k training samples and 17k validation samples.

* How do we validate the human preferences obtained for step 2? If we have 10 different outputs for a prompt, people are likely to rank them in different orders due to their own biases. Thus we can't use a single person's ranking. But how many people do we need for this? Do we use a majority ranking or an average? 

  [InstructGPT (section 5.3)](https://arxiv.org/abs/2203.02155) used around 40 labelers but there was rarely any overlap in the comparisons that they labeled. OpenAI acknowledges that this isn't ideal but does note that the labelers tended to agree with each other roughly 70% of the time.
  
# Open Questions

This section consists of interesting questions that we don't yet have an answer to/are in the process of figuring out.



# Resources

* [wandb rlhf tutorial](https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-trlX--VmlldzozMzAwODM2)
* [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)
* [Hugging Face's StackLLaMa](https://huggingface.co/blog/stackllama)