---
title: "LLM Hyperparameter Tuning Guide: A Comprehensive Analysis from Generation to Deployment"
subtitle: ""
summary: "This article provides an in-depth analysis of two key categories of hyperparameters for large language models (LLMs): generation parameters and deployment parameters, detailing their functions, value ranges, impacts, and best practices across different scenarios to help developers precisely tune models for optimal performance, cost, and output quality."
authors:
- admin
language: en
tags:
- Large Language Models
- Hyperparameter Tuning
- Model Deployment
- API Integration
- vLLM
categories:
- Technical Documentation
- Machine Learning
date: "2025-06-27T03:00:00Z"
lastmod: "2025-06-27T03:00:00Z"
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

## Introduction

Behind the powerful capabilities of large language models (LLMs) is a series of complex hyperparameters working silently. Whether you're deploying a local inference service like vLLM or calling OpenAI's API, precisely tuning these parameters is crucial for achieving ideal performance, cost, and output quality. This document provides a detailed analysis of two key categories of hyperparameters: **Generation (Sampling) Parameters** and **Deployment (Serving) Parameters**, helping you fully master their functions, values, impacts, and best practices across different scenarios.
---

### Part 1: Generation (Sampling) Parameters — Controlling Model Creativity and Determinism

Generation parameters directly control the model's behavior when generating the next token. They primarily revolve around a core question: how to select from thousands of possible next words in the probability distribution provided by the model.

### 1. `temperature`

**In one sentence:** Controls the randomness of generated text. Higher `temperature` increases randomness, making responses more creative and diverse; lower `temperature` decreases randomness, making responses more deterministic and conservative.

*   **Underlying Principle:**
    When generating the next token, the model calculates `logits` (raw, unnormalized prediction scores) for all words in the vocabulary. Typically, we use the `Softmax` function to convert these `logits` into a probability distribution. The `temperature` parameter is introduced before the `Softmax` calculation, "smoothing" or "sharpening" this probability distribution.

    The standard Softmax formula is: `P(i) = exp(logit_i) / Σ_j(exp(logit_j))`

    With `temperature` (T) introduced, the formula becomes: `P(i) = exp(logit_i / T) / Σ_j(exp(logit_j / T))`

    *   When `T` -> 0, the differences in `logit_i / T` become dramatically amplified. The token with the highest logit approaches a probability of 1, while all other tokens approach 0. This causes the model to almost always choose the most likely word, behaving very deterministically and "greedily."
    *   When `T` = 1, the formula reverts to standard Softmax, and the model behaves in its "original" state.
    *   When `T` > 1, the differences in `logit_i / T` are reduced. Tokens with originally lower probabilities get boosted, making the entire probability distribution "flatter." This increases the chance of selecting less common words, introducing more randomness and creativity.

*   **Value Range and Recommendations:**
    *   **Range:** `[0.0, 2.0]` (theoretically can be higher, but OpenAI API typically limits to 2.0).
    *   **`temperature` = 0.0:** Suitable for scenarios requiring deterministic, reproducible, and highly accurate outputs. Examples: code generation, factual Q&A, text classification, data extraction. With identical inputs, outputs will be almost identical (unless the model itself is updated).
    *   **Low `temperature` (e.g., `0.1` - `0.4`):** Suitable for semi-creative tasks requiring rigor and fidelity to source material. Examples: article summarization, translation, customer service bots. Outputs will vary slightly but remain faithful to core content.
    *   **Medium `temperature` (e.g., `0.5` - `0.8`):** A good balance between creativity and consistency, recommended as the default for most applications. Examples: writing emails, marketing copy, brainstorming.
    *   **High `temperature` (e.g., `0.9` - `1.5`):** Suitable for highly creative tasks. Examples: poetry writing, story creation, dialogue script generation. Outputs will be very diverse and sometimes surprising, but may occasionally produce meaningless or incoherent content.

*   **Note:**
    *   It's generally not recommended to modify both `temperature` and `top_p` simultaneously; it's better to adjust just one. OpenAI's documentation explicitly states that modifying only one is typically advised.

### 2. `top_p` (Nucleus Sampling)

**In one sentence:** Controls generation diversity by dynamically determining the sampling pool size through a cumulative probability threshold (`p`) of the highest probability tokens.

*   **Underlying Principle:**
    `top_p` is a more intelligent sampling strategy than `temperature`, also known as **Nucleus Sampling**. Instead of adjusting all token probabilities, it directly defines a "core" candidate set.

    The specific steps are as follows:
    1.  The model calculates the probability distribution for all candidate tokens.
    2.  All tokens are sorted by probability from highest to lowest.
    3.  Starting from the highest probability token, their probabilities are cumulatively added until this sum exceeds the set `top_p` threshold.
    4.  All tokens included in this cumulative sum form the "nucleus" for sampling.
    5.  The model will only sample from this nucleus (typically renormalizing their probabilities), and all other tokens are ignored.

    **Example:** Assume `top_p` = `0.9`.
    *   If the highest probability token "the" has a probability of `0.95`, then the nucleus will contain only "the", and the model will choose it 100%.
    *   If "the" has a probability of `0.5`, "a" has `0.3`, and "an" has `0.1`, then the cumulative probability of these three words is `0.9`. The nucleus will contain {"the", "a", "an"}. The model will sample from these three words according to their (renormalized) probabilities.

*   **Value Range and Recommendations:**
    *   **Range:** `(0.0, 1.0]`.
    *   **`top_p` = 1.0:** Means the model considers all tokens without any truncation (equivalent to no `top_p`).
    *   **High `top_p` (e.g., `0.9` - `1.0`):** Allows for more diverse choices, suitable for creative tasks, similar in effect to higher `temperature`.
    *   **Low `top_p` (e.g., `0.1` - `0.3`):** Greatly restricts the model's range of choices, making its output very deterministic and conservative, similar in effect to extremely low `temperature`.
    *   **General Recommended Value:** `0.9` is a very common default value as it maintains high quality while allowing for some diversity.

*   **`top_p` vs `temperature`:**
    *   `top_p` is more dynamic and adaptive. When the model is very confident about the next step (sharp probability distribution), `top_p` automatically narrows the candidate set, ensuring quality. When the model is less confident (flat distribution), it expands the candidate set, increasing diversity.
    *   `temperature` adjusts the entire distribution "equally," regardless of whether the distribution itself is sharp or flat.
    *   Therefore, `top_p` is generally considered a safer and more robust method for controlling diversity than `temperature`.

### 3. `top_k`

**In one sentence:** Simply and directly samples only from the `k` tokens with the highest probabilities.

*   **Underlying Principle:** This is the simplest truncation sampling method. It directly selects the `k` tokens with the highest probabilities to form the candidate set, then samples from these `k` tokens. All other tokens are ignored.

*   **Value Range and Recommendations:**
    *   **Range:** Integers, such as `1`, `10`, `50`.
    *   **`top_k` = 1:** Equivalent to greedy search, always choosing the most likely word.
    *   **Recommendation:** `top_k` is typically not the preferred sampling strategy because it's too "rigid." In cases where the probability distribution is very flat, it might accidentally exclude many reasonable words; while in cases where the distribution is very sharp, it might include many extremely low-probability, useless words. `top_p` is usually a better choice.

### 4. `repetition_penalty`

**In one sentence:** Applies a penalty to tokens that have already appeared in the context, reducing their probability of being selected again, thereby reducing repetitive content.

*   **Underlying Principle:** After calculating `logits` but before `Softmax`, this parameter iterates through all candidate tokens. If a token has already appeared in the previous context, its `logit` value is reduced (typically divided by the value of `repetition_penalty`).

    `new_logit = logit / penalty` (if token has appeared)
    `new_logit = logit` (if token has not appeared)

    This way, the final probability of words that have already appeared decreases.

*   **Value Range and Recommendations:**
    *   **Range:** `1.0` to `2.0` is common.
    *   **`1.0`:** No penalty applied (default value).
    *   **`1.1` - `1.3`:** A relatively safe range that can effectively reduce unnecessary repetition without overly affecting normal language expression (such as necessary articles like "the").
    *   **Too High Values:** May cause the model to deliberately avoid common words, producing unnatural or even strange sentences.
### 5. `frequency_penalty` & `presence_penalty`

These two parameters are more refined versions of `repetition_penalty`.

*   **`presence_penalty`:**
    *   **Function:** Applies a fixed penalty to all tokens that have **appeared at least once** in the context. It doesn't care how many times the token has appeared; as long as it has appeared, it gets penalized.
    *   **Underlying Principle:** `new_logit = logit - presence_penalty` (if token has appeared at least once).
    *   **Scenario:** This parameter is useful when you want to encourage the model to introduce entirely new concepts and vocabulary, rather than repeatedly discussing topics that have already been mentioned.
    *   **Range:** `0.0` to `2.0`. Positive values penalize new tokens, negative values encourage them.

*   **`frequency_penalty`:**
    *   **Function:** The penalty is proportional to the **frequency** of the token in the context. The more times a word appears, the heavier the penalty it receives.
    *   **Underlying Principle:** `new_logit = logit - count(token) * frequency_penalty`.
    *   **Scenario:** This parameter is effective when you find the model tends to repeatedly use certain specific high-frequency words (even if they are necessary), leading to monotonous language.
    *   **Range:** `0.0` to `2.0`.

*   **Summary:** `presence_penalty` addresses the question of "whether it has appeared," while `frequency_penalty` addresses "how many times it has appeared."

### 6. `seed`

**In one sentence:** By providing a fixed `seed`, you can make the model's output reproducible when other parameters (such as `temperature`) remain the same.

*   **Function:** In machine learning, many operations that seem random are actually "pseudo-random," determined by an initial "seed." Setting the same seed will produce the same sequence of random numbers. In LLMs, this means the sampling process will be completely deterministic.
*   **Scenarios:**
    *   **Debugging and Testing:** When you need to verify whether a change has affected the output, fixing the `seed` can eliminate randomness interference.
    *   **Reproducible Research:** Reproducibility is crucial in academic research.
    *   **Generating Consistent Content:** When you need the model to consistently produce outputs in the same style for the same input.
*   **Note:** For complete reproduction, **all** generation parameters (`prompt`, `model`, `temperature`, `top_p`, etc.) must be identical.

---

### Part 2: Deployment (Serving) Parameters — Optimizing Service Performance and Capacity

Deployment parameters determine how an LLM inference service manages GPU resources, handles concurrent requests, and optimizes overall throughput and latency. These parameters are particularly important in high-performance inference engines like vLLM.

### 1. `gpu_memory_utilization`

**In one sentence:** Controls the proportion of GPU memory that vLLM can use, with the core purpose of reserving space for the **KV Cache**.

*   **Underlying Principle (PagedAttention):**
    The core of vLLM is the PagedAttention mechanism. Traditional attention mechanisms pre-allocate a continuous, maximum-length memory space for each request to store the Key-Value (KV) Cache. This leads to severe memory waste, as most requests are far shorter than the maximum length.

    PagedAttention manages the KV Cache like virtual memory in an operating system:
    1.  It breaks down each sequence's KV Cache into many small, fixed-size "blocks."
    2.  These blocks can be stored non-contiguously in GPU memory.
    3.  A central "Block Manager" is responsible for allocating and releasing these blocks.

    `gpu_memory_utilization` tells vLLM: "You can use this much proportion of the total GPU memory for free management (mainly storing model weights and physical blocks of KV Cache)."

*   **Value Range and Impact:**
    *   **Range:** `(0.0, 1.0]`.
    *   **Default Value:** `0.9` (i.e., 90%).
    *   **Higher Values (e.g., `0.95`):**
        *   **Advantage:** vLLM has more memory for KV Cache, supporting longer contexts and larger batch sizes, thereby increasing throughput.
        *   **Risk:** If set too high, there might not be enough spare memory for CUDA kernels, drivers, or other system processes, easily leading to **OOM (Out of Memory)** errors.
    *   **Lower Values (e.g., `0.8`):**
        *   **Advantage:** Safer, less prone to OOM, reserves more memory for the system and other applications.
        *   **Disadvantage:** Reduced available space for KV Cache, potentially causing vLLM to struggle with high concurrency or long sequence requests, degrading performance. When KV Cache is insufficient, vLLM triggers **Preemption**, swapping out some running sequences and waiting to swap them back in when there's enough space, severely affecting latency. vLLM's warning log `"there is not enough KV cache space. This can affect the end-to-end performance."` is reminding you of this issue.

*   **Recommendations:**
    *   Start with the default value of `0.9`.
    *   If you encounter OOM, gradually lower this value.
    *   If you encounter many preemption warnings and confirm no other processes are occupying large amounts of GPU memory, you can gradually increase this value.

### 2. `max_num_seqs`

**In one sentence:** Limits the maximum number of sequences (requests) that the vLLM scheduler can process **in one iteration (or one batch)**.

*   **Underlying Principle:**
    vLLM's scheduler selects a batch of requests from the waiting queue in each processing cycle. This parameter directly limits the size of this "batch." Together with `max_num_batched_tokens` (which limits the total number of tokens across all sequences in a batch), it determines the scale of batch processing.

*   **Value Range and Impact:**
    *   **Range:** Positive integers, such as `16`, `64`, `256`.
    *   **Higher Values:**
        *   **Advantage:** Allows for higher concurrency, potentially improving GPU utilization and overall throughput.
        *   **Disadvantage:** Requires more intermediate memory (e.g., for storing `logits` and sampling states) and may increase the latency of individual batches. If set too high, even if KV Cache still has space, OOM might occur due to insufficient temporary memory.
    *   **Lower Values:**
        *   **Advantage:** More memory-friendly, potentially lower latency for individual batches.
        *   **Disadvantage:** Limits concurrency capability, potentially leading to underutilization of GPU and decreased throughput.

*   **Recommendations:**
    *   This value needs to be adjusted based on your GPU memory size, model size, and expected concurrent load.
    *   For high-concurrency scenarios, try gradually increasing this value while monitoring GPU utilization and memory usage.
    *   For interactive, low-latency scenarios, consider setting this value lower.
### 3. `max_model_len`

**In one sentence:** Sets the **maximum context length** the model can process (including both prompt and generated tokens).

*   **Underlying Principle:**
    This parameter directly determines how much logical space vLLM needs to reserve for the KV Cache. For example, if `max_model_len` = `4096`, vLLM must ensure its memory management mechanism can support storing KV pairs for up to `4096` tokens per sequence.
    This affects vLLM's memory planning at startup, such as the size of Position Embeddings.

*   **Value Range and Impact:**
    *   **Range:** Positive integers, cannot exceed the maximum length the model was originally trained on.
    *   **Higher Values:**
        *   **Advantage:** Can handle longer documents and more complex contexts.
        *   **Disadvantage:** **Significantly increases** memory consumption. Each token needs to store KV Cache; doubling the length roughly doubles the memory usage. Even if current requests are short, vLLM needs to prepare for potentially long requests, which occupies more KV Cache blocks.
    *   **Lower Values:**
        *   **Advantage:** **Significantly saves** GPU memory. If you know your application scenario will never exceed 1024 tokens, setting this value to 1024 instead of the default 4096 or 8192 will free up a large amount of KV Cache space, supporting higher concurrency.
        *   **Disadvantage:** Any requests exceeding this length will be rejected or truncated.

*   **Recommendations:**
    *   **Set as needed!** This is one of the most effective parameters for optimizing vLLM memory usage. Based on your actual application scenario, set this value to a reasonable maximum with some margin.

### 4. `tensor_parallel_size` & `pipeline_parallel_size`

These two parameters are used for deploying extremely large models across multiple GPUs or nodes.

*   **`tensor_parallel_size`:**
    *   **Function:** Divides **each layer** of the model (such as a large weight matrix) into `N` parts (`N` = `tensor_parallel_size`), placing them on `N` different GPUs. During computation, each GPU only processes its own portion of the data, then exchanges necessary results through high-speed interconnects (like NVLink) via All-Reduce operations, finally merging to get the complete output.
    *   **Scenario:** Used when a single model's volume exceeds the memory of a single GPU. For example, a 70B model cannot fit into a single 40GB A100, but can be deployed across two A100s by setting `tensor_parallel_size=2`.
    *   **Impact:**
        *   **Advantage:** Achieves model parallelism, solving the problem of models not fitting on a single card.
        *   **Disadvantage:** Introduces significant cross-GPU communication overhead, potentially affecting latency. Requires high-speed interconnects between GPUs.

*   **`pipeline_parallel_size`:**
    *   **Function:** Assigns **different layers** of the model to different GPUs or nodes. For example, placing layers 1-10 on GPU 1, layers 11-20 on GPU 2, and so on. Data flows through these GPUs like a pipeline.
    *   **Scenario:** Used when the model is extremely large and needs to be deployed across multiple nodes (machines).
    *   **Impact:**
        *   **Advantage:** Can scale the model to any number of GPUs/nodes.
        *   **Disadvantage:** Creates "pipeline bubbles" as additional overhead, where some GPUs are idle during the start and end phases of the pipeline, reducing utilization.

*   **Combined Use:**
    vLLM supports using both parallelism strategies simultaneously for efficient deployment of giant models on large clusters.

---

### Summary and Best Practices

| Scenario | `temperature` | `top_p` | `repetition_penalty` | `gpu_memory_utilization` | `max_num_seqs` | `max_model_len` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Code Generation/Factual Q&A** | `0.0` - `0.2` | (Not recommended to modify) | `1.0` | `0.9` (Default) | Adjust based on concurrency | Set as needed |
| **Article Summarization/Translation** | `0.2` - `0.5` | (Not recommended to modify) | `1.1` | `0.9` | Adjust based on concurrency | Set to maximum possible document length |
| **General Chat/Copywriting**| `0.7` (Default) | `0.9` (Recommended) | `1.1` - `1.2` | `0.9` | Adjust based on concurrency | Set as needed, e.g., `4096`|
| **Creative Writing/Brainstorming**| `0.8` - `1.2` | `0.95` | `1.0` | `0.9` | Adjust based on concurrency | Set as needed |
| **High Concurrency Throughput Optimization**| (Task dependent) | (Task dependent) | (Task dependent) | Try `0.9` - `0.95` | Gradually increase | Set to the **minimum** value that meets business needs |
| **Low Latency Interaction Optimization** | (Task dependent) | (Task dependent) | (Task dependent) | `0.9` (Default) | Set to lower values (e.g., `16-64`)| Set as needed |
| **Extremely Memory Constrained** | (Task dependent) | (Task dependent) | (Task dependent) | Lower to `0.8` | Set to lower values | Set to the **minimum** value that meets business needs |

**Final Recommendations:**
1.  **Start with Generation Parameters:** First adjust `temperature` or `top_p` to achieve satisfactory output quality.
2.  **Set Deployment Parameters as Needed:** When deploying, first set `max_model_len` to a reasonable minimum value based on your application scenario.
3.  **Monitor and Iterate:** Start with the default `gpu_memory_utilization=0.9` and a moderate `max_num_seqs`. Observe memory usage and preemption situations through monitoring tools (such as `nvidia-smi` and vLLM logs), then gradually adjust these values to find the optimal balance for your specific hardware and workload.