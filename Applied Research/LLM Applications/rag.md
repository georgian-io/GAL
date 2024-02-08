# Information Retrieval / Retrieval Augmented Generation (RAG):

Last Updated: 3rd Jan 2024

## Table of Contents
- [Information Retrieval / Retrieval Augmented Generation (RAG):](#information-retrieval--retrieval-augmented-generation-rag)
  - [Table of Contents](#table-of-contents)
- [1. Introduction](#1-introduction)
  - [1.1. RAG vs Fine-Tuning](#11-rag-vs-fine-tuning)
- [2. Storage](#2-storage)
  - [2.1. Data Ingestion/Indexing](#21-data-ingestionindexing)
  - [2.2. Embedding Models](#22-embedding-models)
  - [2.3. Chunking](#23-chunking)
  - [2.3.1. Determining Chunk Size](#231-determining-chunk-size)
  - [2.3.2. Chunking Strategies](#232-chunking-strategies)
- [3. Retrieval](#3-retrieval)
  - [3.1. Query Transformation](#31-query-transformation)
    - [3.1.1. Query Expansion](#311-query-expansion)
    - [3.1.2. Query Rewriting](#312-query-rewriting)
    - [3.1.3. Query Compression](#313-query-compression)
  - [3.2. Retrieval Strategies](#32-retrieval-strategies)
    - [3.2.1. Semantic Search](#321-semantic-search)
    - [3.2.2. Lexical/Keyword Search](#322-lexicalkeyword-search)
    - [3.2.3. Hybrid Search](#323-hybrid-search)
    - [3.2.4. Metadata Filtering](#324-metadata-filtering)
- [4. Augmentation](#4-augmentation)
- [5. Common Issues](#5-common-issues)
  - [5.1. Semantic/Lexical Search isn't always enough](#51-semanticlexical-search-isnt-always-enough)
  - [5.2. Complex Questions](#52-complex-questions)
  - [5.3. Too many similar documents](#53-too-many-similar-documents)
  - [5.4. Too many relevant documents](#54-too-many-relevant-documents)
  - [5.5. Too few documents](#55-too-few-documents)
  - [5.6. Mismatch between chunks for retrieval \& generation](#56-mismatch-between-chunks-for-retrieval--generation)
- [6. Resources](#6-resources)


# 1. Introduction

In some scenarios, we may want our Large Language Model (LLM) to make use of knowledge that it did not have access to while training. This could be knowledge from private datasets, knowledge not in the training data, or knowledge that did not exist at that time. Providing this knowledge to an LLM could be helpful during the generation process. Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with a text generation model (typically an LLM). 

![RAG System](images/rag_system.jpeg)

Source: [AWS: Retrieval Augmented Generation (RAG)](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html)

The system receives a prompt and a user query. The user query is sent to the retrieval component which retrieves relevant content. The relevant content is appended to the prompt and query and then sent to the LLM to obtain the final answer. Let's look at this with an example. 

We use a preset prompt with variable user queries and context:
```
Given the context: {retrieved_context}
Answer the following question: {user_query}
Solution: 
```

Then, given a user query `What is the capital of Canada?`, we can use our information retrieval system to surface additional context such as `Toronto is the capital of Ontario, and Ottawa is the capital of Canada.`. We can now fill out our complete prompt and send it to the LLM.
```
Given the context: Toronto is the capital of Ontario, and Ottawa is the capital of Canada.
Answer the following question: What is the capital of Canada?
Solution: 
```

There are two primary advantages of using RAG. First, it lets us provide the LLM with knowledge it otherwise would not have access to. This also means that the data we use can be constantly updated with no changes required to the model. Second, it can reduce the risk of hallucination by providing the model with relevant information, as seen in [Retrieval Augmentation Reduces Hallucination in Conversation](https://arxiv.org/abs/2104.07567). 

There are also some scenarios where RAG may not be the best option. For instance, if we want to improve performance on a specific task such as classification or entity extraction, fine-tuning usually does better. The [Georgian LLM Fine-Tuning Hub](https://github.com/georgian-io/LLM-Finetuning-Hub) contains fine-tuning code for several different LLMs as well as benchmarks for their performance on a variety of tasks. We compare the differences between RAG and fine-tuning in the next section.

This document does not examine different vector databases in detail. Rather, it assumes a generic database and considers the design of the RAG pipeline as a whole. The rest of this document covers individual sections of RAG, including storage, retrieval, augmentation, as well as several commonly seen problems.

## 1.1. RAG vs Fine-Tuning
RAG and fine-tuning are often considered as two competing techniques. However they each solve different problems and, in many cases, can be used in conjunction with each other.

Fine-tuning is a technique used to train machine learning models on specific tasks in order to improve their performance on those tasks. At a high level, it can be thought of as a way to let the model learn new patterns and relationships within the type of data it is being fine-tuned on. Fine-tuning usually requires a labeled dataset as well as access to the model's weights.

A simple way of distinguishing their uses is with the phrase "Fine-tuning is for form, not facts" as coined by [AnyScale](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts). That is, fine-tuning is a good choice when we have a narrow and static domain or task. In such scenarios we train the model to reinforce a particular language pattern or style of generation. In contrast, RAG is a good choice when we want to generate text based on facts (where the facts are retrieved context).

In summary, RAG is a good choice if you have a generative task that relies on facts or uses data that is frequently updated. Fine-tuning is a good choice if you have a labeled dataset, access to the model's weights, and want to adapt the model's generation style to a particular dataset. 

# 2. Storage

Storage is a generic term that can refer to any number of data sources. This could include existing databases, APIs, file storage, vector databases etc. In the broadest sense, we would only need a data store and some way of searching for, and retrieving content from the data store. For the purposes of this document, we assume a generic vector database.

A vector or an embedding is a high dimensional representation of data that encapsulates the meaning of a an entity (such as words or phrases). These are created by an embedding model which maps entities into a high dimensional space. A vector database or vectorstore is a data structure that supports the efficient storage of vector embeddings.

[[Back to top]](#)

## 2.1. Data Ingestion/Indexing

In many cases, we might have a collection of data that we want to vectorize and store in a database. This takes place over a few simple steps - extract the data, process or transform the data, and load the processed data into some data structure. Or in other words, Extract-Transform-Load (ETL). 

With vector databases in particular, we would extract data from files and then chunk them into pieces. We perform chunking as documents may be larger than the total context length of the model. In addition, documents may also contain a wide variety of information which may not be relevant to specific tasks. We examine chunking in more detail in a future section. After chunking the documents, we then embed each chunk using an embedding model. Finally, we can store these embeddings in a vector database.

Although we have described this process using embeddings and vector databases, this is not always necessary. Sometimes an existing relational database or API paired with a search query can solve the problem. 

In addition, we can also store additional data alongside text. This metadata can include things like dates, topics, the name of the author etc. We can then use this metadata to help in the retrieval process.

[[Back to top]](#)

## 2.2. Embedding Models

An embedding model has a simple function. Take in some kind of input and output a vector of fixed size that encapsulates the meaning of the input. A good place to start is the [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) which ranks different embedding models based on several criteria.

One question that tends to arise is - why do we use an embedding model that is distinct from the LLM? In general, we tend to use these models as-is for inference with no special fine-tuning applied to them. Thus the main concerns are hosting the model somewhere and inference speed. This is because this model is used not only during the data ingestion process but also during the retrieval process where we embed the user query. Thus we need to find an option that gives us reasonable results while still being relatively inexpensive and fast.

[[Back to top]](#)

## 2.3. Chunking

As described above, chunking is the process of splitting up a document into smaller pieces. We do this so that we can offer the model smaller portions of focused context which may prove more helpful than a large noisy document. There are many factors that can help us determine the specific chunking strategy as well as the size of individual chunks.

[[Back to top]](#)

## 2.3.1. Determining Chunk Size

**LLM Context Size**: We want our chunks to be small enough that we can fit in one or more of them alongside the prompt as context. 

**Expected Queries**: We should take into account the kind of queries we expect. In general, the greater the chunk size, the more high-level information captured. Thus if the expected queries deal with broad and general information, large chunks are helpful. But if the expected queries deal with specific information, smaller chunks are helpful.

**Embedding Model**: We need to also consider the embedding model we use. Some models work well on individual sentences (such as sentence transformers) while other models are designed to work with chunks of specific sizes (such as OpenAI's text-embedding models).

Note: Chunking is still a relatively new technique. The solution might be as simple as testing out a couple of different chunk sizes and seeing what works best for you.

[[Back to top]](#)

## 2.3.2. Chunking Strategies

Once we have a rough idea of the optimal chunk size, we can decide a chunking strategy such as:

* **Fized-size chunking**: This is a simple and inexpensive method of chunking. We just divide the text into chunks of N-tokens. Simply chunking them may however result in chunks losing some context. So in practice, there's usually a small amount of overlap between chunks. This is the approach generally used for OpenAI's text-embedding models.

* **Content-aware chunking**: In this method we chunk based on some idea of what we're chunking. This could be things like sentence-level or paragraph-level chunking.

* **Structure-aware chunking**: Sometimes we work with data that has some kind of known structure - such as Python code or Markdown. We can use this information while chunking. LangChain has an example [here](https://js.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter).

* **Context-aware chunking**: In this scenario we use one of the chunking strategies from above, but in addition, we also store metadata. For instance, if we have were to chunk all of Wikipedia, we might store each article as a chunk and hold metadata on each article. We can then use this in a form of hybrid search during the retrieval phase.

Note: The techniques above are not a comprehensive view of all chunking methods. You can check out the [LangChain documentation](https://js.langchain.com/docs/modules/data_connection/document_transformers/) for more information on other chunking strategies.

[[Back to top]](#)

# 3. Retrieval

Retrieval is the process of acquiring relevant context from the storage given a query from the user. As a reminder, this works through the following steps:

1. Get a query from the user. 
2. Use a retrieval strategy to find relevant text. 
3. Return the K-most relevant chunks.

[[Back to top]](#)

## 3.1. Query Transformation

In many cases we may pass the user query as-is to the retrieval system. However, there are scenarios where we may want to transform the query before sending it to the retrieval system. There are seeveral techniques that may be used here.

### 3.1.1. Query Expansion

Query expansion usually involves decomposing the input into sub-problems, each of which is a more narrow retrieval problem. This can be achieved through prompting techniques such as Step-Back prompting or Least-to-Most prompting, both of which are described in [reasoning.md](reasoning.md)

### 3.1.2. Query Rewriting

In some scenarios, the user query might not be worded well or it might be poorly framed. We can use an LLM to rewrite the query in a better manner. The [Rewrite-Retrieve-Read](https://arxiv.org/abs/2305.14283) framework can be used to achieve this. 

### 3.1.3. Query Compression

Sometimes our query might be quite big or might require some amount of the chat history. However a large query has the drawback of reducing the amount of information we can retrieve as context. As a solution to this, we could take the entire prompt alongside the chat history and create a summary of it using an LLM. In essence, we compress the large chat history and prompt combination to a simple one or two line prompt.

On a similar note, we might also want to retrieve more information than the model's supported context size. In this scenario too, we can use this concept of compression. This involves asking the model to summarize all the retrieved documents. Then this summary can be passed as context instead of a concatenation of the retrieved documents.

[[Back to top]](#)

## 3.2. Retrieval Strategies

There are several different strategies that are employed to find relevant text. In general this means, given some kind of text input, we need to find the K most relevant entities from our storage and return them. K is a value set by the user to determine the maximum number of items to retrieve.

[[Back to top]](#)

### 3.2.1. Semantic Search

A popular approach in the world of LLMs is semantic search (also called dense retrieval).

1. Create an embedding of the text input (using the embedding model).
2. Search the vector store for the K most similar embeddings (via something like cosine similarity).
3. Retrieve the text corresponding to the selected embeddings.
4. Return to the LLM.

Semantic search tends to work well when we need to capture abstract concepts and meanings within the documents we are searching for. It does not do as well when given specific keywords to match (such as IDs). In addition, it can only be used with embeddings.

[[Back to top]](#)

### 3.2.2. Lexical/Keyword Search

Lexical/Keyword search works well when we want to search for specific terms (such as IDs) within documents. It deals with raw text and not embedding data. Thus it may not be as useful as semantic search in capturing abstract concepts. The most popular option for this type of search is [BM25](https://en.wikipedia.org/wiki/Okapi_BM25). [TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) is another option that can be used.

[[Back to top]](#)

### 3.2.3. Hybrid Search

The Hybrid Search method combines several methods of retrieval together. This allows us to gain the advantages of different search strategies such as semantic search and lexical search. However, this does mean we get a number of different results. Thus we need a way to combine these rankings together. This process is known as reranking. A simple reranking method is to just average all the different results. Another more popular option is to use [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). We can think of this as running several algorithms to find a shortlist of potentially good items and then running a final step to filter the shortlist down to K options. For instance, we may want to conduct a search to find documents talking about a concept X but at the same time ensure that it uses a keyword Y.

[[Back to top]](#)

### 3.2.4. Metadata Filtering

While not a standalone search method, we can augment any of the other search methods via metadata filtering. This has a prerequisite of documents being tagged with metadata. Then, we can filter our documents using this metadata before performing the search. For instance, if we have a database of news articles tagged with the type of news, we could filter for "Sports" while search for a query about sports.

# 4. Augmentation

In the augmentation step, we take the retrieved context and augment our prompt to include this context. In most cases, our prompt is mostly static with variables for the user query and retrieved context. For example, the prompt used in the introduction: 

```
Given the context: {retrieved_context}
Answer the following question: {user_query}
Solution: 
```

Ideally, this step should include instructions to the model, either via a system prompt or in the prompt itself. These instructions should specify any rules the LLM should follow, how it should structure its responses, how to handle certain scenarios etc. In any case, the augmented prompt is then sent to the LLM and the corresponding response can then be sent back to the user.

# 5. Common Issues

Now we examine a number of common issues that arise when setting up an information retrieval system alongside an LLM. While we offer a number of solutions to each problem, the recent emergence of this area means that new solutions are still evolving and these solutions may be subject to change.

## 5.1. Semantic/Lexical Search isn't always enough

Sometimes using a single type of search doesn't give us optimal results. For instance, semantic search might not capture specific names or IDs while keyword search may not capture abstract meaning. 

**Solution**: We can use hybrid search methods that combine different information. For instance, we could perform a hybrid search that combines a vector search and BM25. We could use metadata to help in this search.

[[Back to top]](#)

## 5.2. Complex Questions

In some cases we might have complex queries that the retrieval process struggles with. This could be queries that contain multiple questions or it could be complex questions that require answering some component-questions first. In such scenarios, we might want to break them down into individual questions and answer them one at a time.

**Solution**: One option in such a scenario is to simplify the prompts. We could do this manually, or we could use an LLM to break it down into individual questions using query expansion techniques like Least-to-Most prompting or Step-Back prompting. 

[[Back to top]](#)

## 5.3. Too many similar documents

Sometimes we might have a lot of very similar documents in our data. For instance, we could have `report.pdf`, `report1.pdf`, and `report_final.pdf` all containing mostly the same information with small changes between them. In such a scenario, the most relevant documents would just be these nearly-the-same documents, which may not be useful. Thus, we might want a way to either filter these documents.

**Solution**: One solution is to use metadata filtering, though that assumes prior knowledge of similar documents. Another solution is to retrieve a larger than usual number of documents and then perform a filtering step on them. This filter could be something like picking a document and removing its N closest neighbors from the pool, or a clustering algorithm where we pick document centers, or maybe just a simple minimum distance being enforced between retrieved embeddings.

[[Back to top]](#)

## 5.4. Too many relevant documents

If the query is a popular term or just too broad, then we might end up getting tens or even hundreds of very relevant documents. This may not be very desirable, especially if the results are very close to each other in terms of relevance. How do we compare or rank a large number of such documents? 

**Solution**: The obvious answer is to just be more specific. This can also include using hybrid search or metadata filtering. One other option is to select a large number of results and use an LLM to summarize it.

[[Back to top]](#)

## 5.5. Too few documents

In some cases, we might not find enough relevant documents. Or we might have a scenario where we want all potentially relevant documents, even if it means we retrieve some irrelevant or not useful documents. In such a case where we need high recall over everything else, what can we do?

**Solution**: We can broaden our search, use hybrid search, metadata filtering or, in the case of keyword search, we can use our LLM to create variations of the question.

[[Back to top]](#)

## 5.6. Mismatch between chunks for retrieval & generation

Smaller chunks do help improve the search experience but do not necessarily help us with the actual generation part. That is, while generating we might want to have more context than, say, a single sentence. 

**Solution**: We can approach this by decoupling the two. That is, embeddings could be tid to larger or smaller contexts depending on the task. For instance, we could embed individual sentences but associate it with the entire paragraph. This is known as [Small-to-Big Retrieval](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4).

[[Back to top]](#)

# 6. Resources

Here we list resources that were useful in the creation of this document.

* [LangChain Documentation](https://python.langchain.com/docs/get_started)
* [Beware Tunnel Vision in AI Retrieval](https://colinharman.substack.com/p/beware-tunnel-vision-in-ai-retrieval)
* [Pinecone: Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)

[[Back to top]](#)