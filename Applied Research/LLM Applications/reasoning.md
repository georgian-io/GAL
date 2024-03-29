# Reasoning

## Table of Contents
- [Reasoning](#reasoning)
  - [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Chain-of-Thought (CoT)](#chain-of-thought-cot)
  - [Zero-Shot CoT](#zero-shot-cot)
  - [Few-Shot CoT](#few-shot-cot)
  - [Automatic Chain-of-Thought (Auto-CoT)](#automatic-chain-of-thought-auto-cot)
  - [Multimodal Chain-of-Thought](#multimodal-chain-of-thought)
- [Self-Consistency](#self-consistency)
- [Knowledge Generation](#knowledge-generation)
- [Least-to-Most Prompting](#least-to-most-prompting)
- [Step-Back Prompting](#step-back-prompting)
- [Agents](#agents)
  - [Tool Use](#tool-use)
  - [ReAct](#react)
  - [Plan-and-Solve](#plan-and-solve)
- [Resources](#resources)
  - [References](#references)

# Introduction

LLMs can be useful for a variety of tasks. Sometimes though, they need a little nudging in the right direction to get the desired outcome. Broadly, reasoning refers to a collection of techniques that can perform this nudge. In essence, we want to make the model "think" through things, break a task down into individual steps or explain its own reasoning.

Note: When we say reason or think in this context, we are not referring to the kind of thinking or reasoning that we, as humans, are capable of. This [blog](https://cacm.acm.org/blogs/blog-cacm/276268-can-llms-really-reason-and-plan/fulltext) encapsulates the difference between these two ideas.

# Chain-of-Thought (CoT)

Chain-of-Thought is a popular approach to making a model think. The idea is to make the model break down the intermediate steps between the input and the output. Consider the following scenario where we ask the model a question:

![CoT Base Case](images/cot_base_case.png)

We can see that the model contradicts itself by giving the wrong answer but a correct explanation.

Note: This example and all other examples were created using the [OpenAI playground](https://platform.openai.com/playground). We used `gpt-3.5-turbo` (as of September 5th 2023) with default parameters. These examples are illustrative and may no longer work on updated versions of the model.

## Zero-Shot CoT

A simple way to perform CoT is by appending the phrase "Let's think step by step" to the prompt. Taking the same example as above:

![CoT Zero-Shot](images/cot_zero_shot.png)

We see that the model explains its reasoning and then provides the correct answer. This method is known as zero-shot CoT because we do not give the model any prior examples to learn from. That is, the model has had zero shots at this question before. We use the phrase "Let's think step by step" over other similar sounding phrases as some [research](https://arxiv.org/abs/2205.11916) has shown that it is effective. 

## Few-Shot CoT

In some cases, zero-shot CoT may not be successful. This could be due to the complexity of the task. Few-shot CoT attempts to rectify this by providing the model with some examples that illustrate how to perform the task. Like so:

![CoT Few-Shot](images/cot_few_shot.png)

Here we provided a single example on how to handle a similar task. The model learns from this example while attempting to answer our question.

## [Automatic Chain-of-Thought (Auto-CoT)](https://arxiv.org/abs/2210.03493)

Few-Shot CoT requires some effort in finding or crafting examples. Auto-CoT is an approach that eliminates this manual effort by leveraging zero-shot CoT. More specifically, Auto-CoT uses two techniques: question clustering and demonstration sampling. First, it partitions questions in a dataset into a few clusters. Following this, demonstration sampling is performed. In the demonstration sampling process, a representative question is picked from each cluster. Questions within each cluster are sorted by their closeness to the cluster center. Zero-shot CoT is used to generate answers to questions. The first question-answer pair that meets some criteria (such as total length or number of reasoning steps) is selected as the representative of that cluster.

![Auto-CoT](images/cot_auto.png)

Source: [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)

## [Multimodal Chain-of-Thought](https://arxiv.org/abs/2302.00923)

CoT usually focuses only on text but can also be extended to a combination of images and text. The idea is to fuse the image embeddings (obtained through a vision transformer) with the text embeddings and then decode the resulting output. Multimodal CoT happens over two steps. In the first step, the text and image input embeddings are fused and then passed to the LLM to generate a rationale. Then, this rationale is appended to the original text input. This new text input and the original image information are once again fused and passed to the LLM to generate the final answer. 

The example below demonstrates this in action. However, it is important to remember that the performance of such a system is dependent on the vision model. In addition, fine-tuning might also be required to align the vision embeddings with the text embeddings.

![Multimodal CoT](images/multimodal_cot.png)

Source: [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923)

# [Self-Consistency](https://arxiv.org/abs/2203.11171)

In some scenarios, we may observe that the LLM doesn't always return the same answer to a given question. In other scenarios, it may hallucinate. One solution to these problems is to enforce consistency on the output of the LLM. Or rather, we generate multiple answers and take the most consistent answer. In a simple case, we get some N answers and take the majority vote as our final answer. Self-consistency is typically viable in scenarios where we expect some kind of structured output so that we can assess the answer.

Self-consistency can be a good tool to use for a couple of reasons. First, it's unsupervised, with no need for human intervention. Second, it requires no additional training, or fine-tuning. From a more traditional machine learning perspective, self-consistency can be thought of as a self-ensemble of models i.e., an ensemble of models where we use the same model each time.

Consider the following example:

![Self-Consistency](images/self_consistency.png)

Here we see that the first model output is incorrect. However, if we take the majority vote here, we reach the correct outcome. 

In scenarios where we may not have a structured output or otherwise have an answer that isn't easily extractable, self-consistency doesn't work. In these scenarios, one alternative may be to use another, more powerful LLM to act as a judge of the outputs. As an example, we could use GPT-3.5 to generate 10 answers and then prompt GPT-4 to judge an optimal answer.

One caveat with self-consistency is that it can be an expensive process. Instead of making a single call for a prompt, we end up making several. This process can add up over time, especially when we scale things up. Hence, self-consistency may not always be an appropriate choice. As an alternative, many LLMs become effectively deterministic when the temperature is set to 0 (with the exception of GPT-4). This usually comes at a cost of reduced creativity from the LLM - which may or may not be useful depending on the scenario.

# Knowledge Generation

There are some scenarios where we might want the LLM to reference some knowledge while generating text. Usually, this involves the use of external knowledge sources such as private documents or external sources like Wikipedia. We cover this topic in the `information_retrieval.md` document. In other scenarios, we might want to rely on knowledge that the model itself produces. In such cases, we ask the model to generate the knowledge for us. While asking the model to generate knowledge poses a risk of hallucination, it can help guide the model in terms of the answer it generates. 

For instance:

![Knowledge Generation](images/knowledge_generation.png)

Note: the default maximum length was 256 tokens, which is why the second paragraph cuts off.

An alternative approach may be to split up the prompt to ask for facts separately and then pass the facts as part of the second prompt. In this scenario, we would not encounter the token limit issue.

# [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625)

Least-to-Most prompting is a way to get the model to break down tasks into a set of subtasks. Then each subtask is solved in order with the prompt and solution for the previous subtask being fed into the model at each step. In essence, we try to solve smaller, individual problems and slowly build our way to the final answer. The example below demonstrates this two-step process.

![Least-to-Most Prompting](images/least_to_most.png)

Source: [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625)

Generally, least-to-Most prompting can work well but may run into problems with context size due to the addition of each subtask to the prompt. These problems may be more likely  when the initial prompt is large or if there are a lot of subtasks. One alternative to this method is to use agents, that is, LLMs that can make use of reasoning and tools. 

# [Step-Back Prompting](https://arxiv.org/abs/2310.06117)

Step-Back prompting involves "taking a step back" when faced with a task. In essence, when given a question that may contain a lot of details, we take a step back and distill it into a higher level abstraction that encompasses the original question. For instance, given a question about the school some one went to in a particular time span, we can distill it into a more abstract "education history". We can then use the model's response to this question to ground the answer it provides (by passing it as context) to the original question.

![Step-Back Prompting](images/step_back.png)

Source: [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117)

As the example above demonstrates, step-back prompting can sometimes succeed where regular CoT fails. By explicitly asking the model to take a step back and answer a more fundamental question, we can equip it with the means of solving the task at hand.

# Agents

So far we have seen different methods of making an LLM "think" or "reason".
An agent can be thought of as an LLM that has access to a collection of tools (called a toolkit).

## Tool Use

For some tasks, we may not need to use an LLM. For instance, we already know how to do math. In this case, we don't need to ask the LLM to sum the rows of a spreadsheet. Instead, we might want to use the spreadsheet's existing summation capabilities. In other cases, we might not be able to complete a task with just an LLM. For instance, if we wanted to find and summarize news articles on some recent event, we would need to provide the LLM with the capability to search. 

Tools can thus be thought of as augmentations to enhance an LLM's capabilities. There are tools to search Google, perform complex math, interface with APIs, interact with databases and more. In order to use these tools, we provide a text description of the available tools in our initial prompt alongside the function definition. The LLM can then "call" the tool (usually handled by libraries like Langchain) with the correct parameters to obtain the result of that function.

So far, we have seen ways to make the LLM reason. Tools can be powerful in that they enable the LLM to act on their thoughts. This combination of reasoning and tools is the foundation for agent frameworks like ReAct and Plan-and-Solve. 

## [ReAct](https://arxiv.org/abs/2210.03629)

ReAct or "Reasoning and Acting" is a framework through which we make an LLM reason about the task and then perform actions based on it. This framework allows the model to not only create plans of action but also track and update their status. Here, reasoning refers to the various thought methods we discussed above such as CoT, while acting refers to the actions an agent takes based on the reasoning step. Acting usually includes the use of tools. 

Consider the following prompt:
```
Question: Aside from the Apple Remote, what other device can control the program the Apple Remote was originally designed to interact with?
```

This example is a complex prompt, which does not have an obvious answer. The LLM might not even have knowledge about the program or the device in question. Thus, this problem cannot be solved using just reasoning. Acting alone might not ask the right questions. Instead, we do a combination of both in the example below:

![ReAct](images/react.png)

Source: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

## [Plan-and-Solve](https://arxiv.org/abs/2305.04091)

Plan-and-Solve (sometimes referred to as plan-and-execute) prompting is a framework through which we ask an LLM to plan out the subtasks required to complete a task, and then carry out the subtasks according to the plan. In this framework there are actually two entities, a planner and a solver. The planner is usually an LLM while the solver is another LLM (usually just the same LLM but treated independently). In essence, the planner breaks down the task while the solver completes each subtask. An advantage of this strategy is that it keeps the planning and execution prompts separate, thus allowing the LLM to focus on a single problem at a time.

The planner LLM is prompted to break down a task into subtasks similar to Least-to-Most prompting while the solver LLM uses a framework like ReAct to determine how to complete a subtask.

![Plan-and-Solve](images/plan_and_solve.png)

Source: [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091)

# Resources

* [LangChain: Documentation](https://python.langchain.com/docs/get_started/introduction.html)
* [ACM: Can LLMs Really Reason and Plan?](https://cacm.acm.org/blogs/blog-cacm/276268-can-llms-really-reason-and-plan/fulltext)

## References

The following is a list of papers that cover the majority of the techniques we've discussed in this document.

* [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
* [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)
* [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923)
* [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
* [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
* [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625)
* [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091)