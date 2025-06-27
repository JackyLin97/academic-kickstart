---
title: "Ollama Practical Guide: Local Deployment and Management of Large Language Models"
subtitle: ""
summary: "This article provides a detailed introduction to Ollama, a powerful open-source tool, covering its core concepts, quick start guide, API reference, command-line tools, and advanced features, helping users easily download, run, and manage large language models in local environments."
authors:
- admin
language: en
tags:
- Large Language Models
- Local Deployment
- API
- Open Source Tools
- Model Management
categories:
- Technical Documentation
- Artificial Intelligence
date: "2025-06-27T02:00:00Z"
lastmod: "2025-06-27T02:00:00Z"
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

## 1. Introduction

Ollama is a powerful open-source tool designed to allow users to easily download, run, and manage large language models (LLMs) in local environments. Its core advantage lies in simplifying the deployment and use of complex models, enabling developers, researchers, and enthusiasts to experience and utilize state-of-the-art artificial intelligence technology on personal computers without specialized hardware or complex configurations.

**Key Advantages:**

*   **Ease of Use:** Complete model download, running, and interaction through simple command-line instructions.
*   **Cross-Platform Support:** Supports macOS, Windows, and Linux.
*   **Rich Model Library:** Supports numerous popular open-source models such as Llama 3, Mistral, Gemma, Phi-3, and more.
*   **Highly Customizable:** Through `Modelfile`, users can easily customize model behavior, system prompts, and parameters.
*   **API-Driven:** Provides a REST API for easy integration with other applications and services.
*   **Open Source Community:** Has an active community continuously contributing new models and features.

This document will provide a comprehensive introduction to Ollama's various features, from basic fundamentals to advanced applications, helping you fully master this powerful tool.

---

## 2. Quick Start

This section will guide you through installing and basic usage of Ollama.

### 2.1 Installation

Visit the [Ollama official website](https://ollama.com/) to download and install the package suitable for your operating system.

### 2.2 Running Your First Model

After installation, open a terminal (or command prompt) and use the `ollama run` command to download and run a model. For example, to run the Llama 3 model:

```shell
ollama run llama3
```

On first run, Ollama will automatically download the required model files from the model library. Once the download is complete, you can directly converse with the model in the terminal.

### 2.3 Managing Local Models

You can use the following commands to manage locally downloaded models:

*   **List Local Models:**
    ```shell
    ollama list
    ```
    This command displays the name, ID, size, and modification time of all downloaded models.

*   **Remove Local Models:**
    ```shell
    ollama rm <model_name>
    ```

---

## 3. Core Concepts

### 3.1 Modelfile

`Modelfile` is one of Ollama's core features. It's a configuration file similar to `Dockerfile` that allows you to define and create custom models. Through `Modelfile`, you can:

*   Specify a base model.
*   Set model parameters (such as temperature, top_p, etc.).
*   Define the model's system prompt.
*   Customize the model's interaction template.
*   Apply LoRA adapters.

A simple `Modelfile` example:

```Modelfile
# Specify base model
FROM llama3

# Set model temperature
PARAMETER temperature 0.8

# Set system prompt
SYSTEM """
You are a helpful AI assistant. Your name is Roo.
"""
```

Use the `ollama create` command to create a new model based on a `Modelfile`:

```shell
ollama create my-custom-model -f ./Modelfile
```

### 3.2 Model Import

Ollama supports importing models from external file systems, particularly from `Safetensors` format weight files.

In a `Modelfile`, use the `FROM` directive and provide the directory path containing `safetensors` files:

```Modelfile
FROM /path/to/safetensors/directory
```

Then use the `ollama create` command to create the model.

### 3.3 Multimodal Models

Ollama supports multimodal models (such as LLaVA) that can process both text and image inputs simultaneously.

```shell
ollama run llava "What's in this image? /path/to/image.png"
```

---

## 4. API Reference

Ollama provides a set of REST APIs for programmatically interacting with models. The default service address is `http://localhost:11434`.

### 4.1 `/api/generate`

Generate text.

*   **Request (Streaming):**
    ```shell
    curl http://localhost:11434/api/generate -d '{
      "model": "llama3",
      "prompt": "Why is the sky blue?"
    }'
    ```
*   **Request (Non-streaming):**
    ```shell
    curl http://localhost:11434/api/generate -d '{
      "model": "llama3",
      "prompt": "Why is the sky blue?",
      "stream": false
    }'
    ```

### 4.2 `/api/chat`

Conduct multi-turn conversations.

*   **Request:**
    ```shell
    curl http://localhost:11434/api/chat -d '{
      "model": "llama3",
      "messages": [
        {
          "role": "user",
          "content": "why is the sky blue?"
        }
      ],
      "stream": false
    }'
    ```

### 4.3 `/api/embed`

Generate embedding vectors for text.

*   **Request:**
    ```shell
    curl http://localhost:11434/api/embed -d '{
      "model": "all-minilm",
      "input": ["Why is the sky blue?", "Why is the grass green?"]
    }'
    ```

### 4.4 `/api/tags`

List all locally available models.

*   **Request:**
    ```shell
    curl http://localhost:11434/api/tags
    ```

---

## 5. Command Line Tools (CLI)

Ollama provides a rich set of command-line tools for managing models and interacting with the service.

*   `ollama run <model>`: Run a model.
*   `ollama create <model> -f <Modelfile>`: Create a model from a Modelfile.
*   `ollama pull <model>`: Pull a model from a remote repository.
*   `ollama push <model>`: Push a model to a remote repository.
*   `ollama list`: List local models.
*   `ollama cp <source_model> <dest_model>`: Copy a model.
*   `ollama rm <model>`: Delete a model.
*   `ollama ps`: View running models and their resource usage.
*   `ollama stop <model>`: Stop a running model and unload it from memory.

---

## 6. Advanced Features

### 6.1 OpenAI API Compatibility

Ollama provides an endpoint compatible with the OpenAI API, allowing you to seamlessly migrate existing OpenAI applications to Ollama. The default address is `http://localhost:11434/v1`.

*   **List Models (Python):**
    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )

    response = client.models.list()
    print(response)
    ```

### 6.2 Structured Output

By combining the OpenAI-compatible API with Pydantic, you can force the model to output JSON with a specific structure.

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

class UserInfo(BaseModel):
    name: str
    age: int

try:
    completion = client.beta.chat.completions.parse(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": "My name is John and I am 30 years old."}],
        response_format=UserInfo,
    )
    print(completion.choices[0].message.parsed)
except Exception as e:
    print(f"Error: {e}")
```

### 6.3 Performance Tuning

You can adjust Ollama's performance and resource management through environment variables:

*   `OLLAMA_KEEP_ALIVE`: Set how long models remain active in memory. For example, `10m`, `24h`, or `-1` (permanent).
*   `OLLAMA_MAX_LOADED_MODELS`: Maximum number of models loaded into memory simultaneously.
*   `OLLAMA_NUM_PARALLEL`: Number of requests each model can process in parallel.

### 6.4 LoRA Adapters

Use the `ADAPTER` directive in a `Modelfile` to apply a LoRA (Low-Rank Adaptation) adapter, changing the model's behavior without modifying the base model weights.

```Modelfile
FROM llama3
ADAPTER /path/to/your-lora-adapter.safetensors
```

---

## 7. Appendix

### 7.1 Troubleshooting

*   **Check CPU Features:** On Linux, you can use the following command to check if your CPU supports instruction sets like AVX, which are crucial for the performance of certain models.
    ```shell
    cat /proc/cpuinfo | grep flags | head -1
    ```

### 7.2 Contribution Guidelines

Ollama is an open-source project, and community contributions are welcome. When submitting code, please follow good commit message formats, for example:

*   **Good:** `llm/backend/mlx: support the llama architecture`
*   **Bad:** `feat: add more emoji`

### 7.3 Related Links

*   **Official Website:** [https://ollama.com/](https://ollama.com/)
*   **GitHub Repository:** [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
*   **Model Library:** [https://ollama.com/library](https://ollama.com/library)