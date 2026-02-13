# Ahoum-AI-Task
## 1. Problem Statement Description

The goal of this assignment is to design a production-ready conversation evaluation benchmark that scores each conversation turn across a large number of distinct evaluation facets. These facets span multiple dimensions including:

* Linguistic quality

* Pragmatics and intent

* Emotional tone

* Behavioral and safety-related signals

The benchmark must:

* Handle 300 facets, with architectural support for scaling to 5000+ facets

* Assign ordinal scores (0–5) per facet

* Avoid one-shot prompt solutions

* Use open-weight models ≤ 16B parameters

* Be robust, modular, and extensible

This repository implements an end-to-end, scalable evaluation pipeline that satisfies all core requirements while remaining efficient and interpretable.

## 2. End-to-End Workflow

**Step 0: Install Dependencies**

Before running the pipeline, install the required libraries:

```bash
pip install -r requirements.txt
```
This installs:
* ```chromadb```
* ```bitsandbytes```
* ```torch```
* ```transformers```

**Step 1: Load and Clean Facets**

* The provided CSV contains 300 evaluation facets

* Each facet is cleaned using regular expressions (re)

* Cleaning removes:

  * Leading numbering

  * Special characters

  * Extra whitespace

  * Case inconsistencies

This ensures semantic consistency before embedding.

**Step 2: Embed Facets and Store in Vector Database**

* Cleaned facets are embedded using **BGE-M3**

* Embeddings are stored in **ChromaDB**

* Each entry stores:

  * Vector embedding

  * Cleaned facet text

  * Original facet label (metadata)

This allows fast semantic retrieval at inference time.

**Step 3: Analyze Conversation Text**

* Each conversation (single-turn or multi-turn) is passed into the system

* Conversations are treated as atomic evaluation units

* The system does not rely on handcrafted rules

**Step 4: Facet Retrieval (Router Agent)**

* A retrieval agent (router_agent) embeds the conversation text

* ChromaDB is queried to retrieve top-10 most relevant facets

* These facets are passed downstream for scoring

This enforces facet sparsity, reducing noise and hallucination.

**Step 5: Facet Scoring with Mistral-7B**

* The retrieved facets and conversation text are passed to Mistral-7B

* The model assigns ordinal scores (0–5) per facet

* Output is post-processed and saved as structured JSON

**Step 6: JSON Output Generation**

The final output is a JSON file containing:

* Conversation ID

* Conversation text

* Number of facets scored

* List of ```(facet, score)``` pairs


## 3. Why ChromaDB and BGE-M3?

### Why ChromaDB?

* Lightweight, local-first vector database

* No external service dependency

* Efficient approximate nearest neighbor (HNSW)

* Seamlessly scales to thousands of facets

ChromaDB is ideal for retrieval-augmented evaluation pipelines.

### Why BGE-M3 for Embeddings?

* Strong performance on semantic retrieval

* Handles short phrases and abstract traits well

* Open-weight and efficient

* Consistent embeddings across domains

BGE-M3 ensures relevant facet retrieval even under ambiguity.

## 4. Conversation Text Analysis (Sample Output)

### Input Conversation

```bash
I’m not sure this solution will work in production.
```
### Output (Excerpt)
```bash
  {
    "conversation_id": 1,
    "conversation_text": "I’m not sure this solution will work in production.",
    "num_facets_scored": 10,
    "facet_scores": [
      {
        "facet": "Troubleshooting technical issues",
        "score": 4
      },
      {
        "facet": "Impracticalness",
        "score": 3
      },
      {
        "facet": "Evaluating Solutions",
        "score": 2
      },
      {
        "facet": "Inefficiency",
        "score": 1
      },
      {
        "facet": "Seeking approval",
        "score": 0
      },
      {
        "facet": "Creativity in solutions",
        "score": 0
      },
      {
        "facet": "Common-sense",
        "score": 0
      },
      {
        "facet": "Harmfulness",
        "score": 0
      },
      {
        "facet": "Discontentment",
        "score": 0
      },
      {
        "facet": "Hesitation",
        "score": 0
      }
    ]
  }
```
The system correctly identifies:

* Problem

* Inefficiency

And also assign 0 to non-related facets

## 5. Facet Cleaning, Embedding, and Storage

### Cleaning with re

Regular expressions are used to:

* Remove numbering artifacts

* Normalize text

* Improve embedding quality

Example:

```bash
re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
```

### Embedding with BGE-M3

* Cleaned facets are embedded once

* Embeddings are reused across all conversations

* This significantly improves efficiency

### Storage in ChromaDB

* Enables fast similarity search

* Supports metadata storage

* Allows future expansion without code changes

## 6. Retriever Agent (Router Agent) and Scalability

### Router Agent Design

The retriever agent (```router_agent```) is responsible for:

* Embedding conversation text

* Querying ChromaDB

* Returning the top-10 most relevant facets

This agent acts as an **AI-based filter**, ensuring only meaningful facets are scored.

### Why Scalability Is Guaranteed

Whether the system has:

* 300 facets

* 5000 facets

* 50000 facets

…the retriever always selects only the top-10 relevant facets.

This means:

* Runtime remains constant

* No architectural redesign is required

* The benchmark naturally scales

This design directly satisfies the **assignment’s scalability constraint**.

## 7. Why Mistral-7B Instead of Llama-3-8B?

* Llama-3-8B required special access approval

* Mistral-7B is:

  * Open-weight

  * Well-aligned

  * Efficient under quantization

  * Strong at instruction-following

### Scoring Capability

Mistral-7B:

* Understands abstract facets

* Assigns consistent ordinal scores

* Handles short conversational text well

### Output Format

* Scores are extracted and normalized

* Final output is written as structured JSON

## 8. Challenges Faced

### 1. Hallucination in Mistral-7B

The model sometimes generated:

* Unrelated explanations

* Documentation-style text

* Flask or API examples

Solution:

* Clipped output using regex

* Extracted only the first valid numeric sequence

* Ensured clean JSON output

### 2. GPU Constraints

* CPU inference was extremely slow

* Local laptop GPU was not properly CUDA-enabled

Solution:

* Used Google Colab free GPU

* Applied 4-bit quantization via bitsandbytes

* Balanced performance and memory usage

## 9. Conclusion and Future Work

### Conclusion

This project delivers a:

* Scalable

* Modular

* Production-ready

* Retrieval-augmented conversation evaluation benchmark

It satisfies all assignment constraints and demonstrates robust system design.

### Future Work

Planned improvements include:

* Upgrading to higher-parameter models

* Renting dedicated GPUs for faster inference

* Adding confidence scores

* Generating rationales explaining why each score was assigned

* Supporting multi-annotator comparison

## Final Note

This repository demonstrates that sparse, retrieval-driven evaluation is more robust and scalable than brute-force scoring, making it suitable for real-world conversational AI benchmarking.
