# Ollama 技术文档

## 1. 简介

Ollama 是一个强大的开源工具，旨在让用户能够轻松地在本地环境下载、运行和管理大型语言模型（LLM）。它的核心优势在于简化了部署和使用复杂模型的流程，使得开发者、研究人员和爱好者无需专业的硬件或复杂的配置，即可在个人计算机上体验和利用 state-of-the-art 的人工智能技术。

**主要优势:**

*   **易于使用:** 通过简单的命令行指令，即可完成模型的下载、运行和交互。
*   **跨平台支持:** 支持 macOS, Windows, 和 Linux。
*   **模型库丰富:** 支持众多流行的开源模型，如 Llama 3, Mistral, Gemma, Phi-3 等。
*   **高度可定制:** 通过 `Modelfile`，用户可以轻松地自定义模型的行为、系统提示和参数。
*   **API 驱动:** 提供 REST API，方便与其他应用程序和服务集成。
*   **开源社区:** 拥有活跃的社区，不断贡献新的模型和功能。

本篇文档将深入浅出地介绍 Ollama 的各项功能，从基础入门到高级应用，帮助您全面掌握这个强大的工具。

---

## 2. 快速入门

本节将指导您完成 Ollama 的安装和基本使用。

### 2.1 安装

访问 [Ollama 官方网站](https://ollama.com/) 下载适用于您操作系统的安装包并进行安装。

### 2.2 运行第一个模型

安装完成后，打开终端（或命令提示符），使用 `ollama run` 命令来下载并运行一个模型。例如，运行 Llama 3 模型：

```shell
ollama run llama3
```

首次运行时，Ollama 会自动从模型库下载所需的模型文件。下载完成后，您就可以直接在终端与模型进行对话。

### 2.3 管理本地模型

您可以使用以下命令来管理本地已下载的模型：

*   **列出本地模型:**
    ```shell
    ollama list
    ```
    该命令会显示所有已下载模型的名称、ID、大小和修改时间。

*   **移除本地模型:**
    ```shell
    ollama rm <model_name>
    ```

---

## 3. 核心概念

### 3.1 Modelfile

`Modelfile` 是 Ollama 的核心功能之一，它是一个类似于 `Dockerfile` 的配置文件，允许您定义和创建自定义模型。通过 `Modelfile`，您可以：

*   指定基础模型。
*   设置模型参数（如温度、top_p 等）。
*   定义模型的系统提示（System Prompt）。
*   自定义模型的交互模板。
*   应用 LoRA 适配器。

一个简单的 `Modelfile` 示例：

```Modelfile
# 指定基础模型
FROM llama3

# 设置模型温度
PARAMETER temperature 0.8

# 设置系统提示
SYSTEM """
You are a helpful AI assistant. Your name is Roo.
"""
```

使用 `ollama create` 命令基于 `Modelfile` 创建新模型：

```shell
ollama create my-custom-model -f ./Modelfile
```

### 3.2 模型导入

Ollama 支持从外部文件系统导入模型，特别是从 `Safetensors` 格式的权重文件。

在 `Modelfile` 中，使用 `FROM` 指令并提供包含 `safetensors` 文件的目录路径：

```Modelfile
FROM /path/to/safetensors/directory
```

然后使用 `ollama create` 命令创建模型。

### 3.3 多模态模型

Ollama 支持多模态模型（如 LLaVA），可以同时处理文本和图像输入。

```shell
ollama run llava "这张图片里有什么? /path/to/image.png"
```

---

## 4. API 参考

Ollama 提供了一套 REST API，用于以编程方式与模型进行交互。默认服务地址为 `http://localhost:11434`。

### 4.1 `/api/generate`

生成文本。

*   **请求 (Streaming):**
    ```shell
    curl http://localhost:11434/api/generate -d '{
      "model": "llama3",
      "prompt": "Why is the sky blue?"
    }'
    ```
*   **请求 (Non-streaming):**
    ```shell
    curl http://localhost:11434/api/generate -d '{
      "model": "llama3",
      "prompt": "Why is the sky blue?",
      "stream": false
    }'
    ```

### 4.2 `/api/chat`

进行多轮对话。

*   **请求:**
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

生成文本的嵌入向量。

*   **请求:**
    ```shell
    curl http://localhost:11434/api/embed -d '{
      "model": "all-minilm",
      "input": ["Why is the sky blue?", "Why is the grass green?"]
    }'
    ```

### 4.4 `/api/tags`

列出本地所有可用的模型。

*   **请求:**
    ```shell
    curl http://localhost:11434/api/tags
    ```

---

## 5. 命令行工具 (CLI)

Ollama 提供了一套丰富的命令行工具来管理模型和与服务交互。

*   `ollama run <model>`: 运行一个模型。
*   `ollama create <model> -f <Modelfile>`: 从 Modelfile 创建一个模型。
*   `ollama pull <model>`: 从远程库拉取一个模型。
*   `ollama push <model>`: 将一个模型推送到远程库。
*   `ollama list`: 列出本地模型。
*   `ollama cp <source_model> <dest_model>`: 复制一个模型。
*   `ollama rm <model>`: 删除一个模型。
*   `ollama ps`: 查看正在运行的模型及其资源占用。
*   `ollama stop <model>`: 停止一个正在运行的模型并将其从内存中卸载。

---

## 6. 高级功能

### 6.1 OpenAI API 兼容性

Ollama 提供了一个与 OpenAI API 兼容的端点，允许您将现有的 OpenAI 应用无缝迁移到 Ollama。默认地址为 `http://localhost:11434/v1`。

*   **列出模型 (Python):**
    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )

    response = client.models.list()
    print(response)
    ```

### 6.2 结构化输出

结合使用 OpenAI 兼容 API 和 Pydantic，可以强制模型输出特定结构的 JSON。

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

### 6.3 性能调优

您可以通过环境变量来调整 Ollama 的性能和资源管理：

*   `OLLAMA_KEEP_ALIVE`: 设置模型在内存中保持活动状态的时间。例如 `10m`, `24h`, 或 `-1` (永久)。
*   `OLLAMA_MAX_LOADED_MODELS`: 同时加载到内存中的最大模型数量。
*   `OLLAMA_NUM_PARALLEL`: 每个模型可以并行处理的请求数量。

### 6.4 LoRA 适配器

在 `Modelfile` 中使用 `ADAPTER` 指令来应用一个 LoRA (Low-Rank Adaptation) 适配器，从而在不修改基础模型权重的情况下，改变模型的行为。

```Modelfile
FROM llama3
ADAPTER /path/to/your-lora-adapter.safetensors
```

---

## 7. 附录

### 7.1 故障排除

*   **检查 CPU 特性:** 在 Linux 上，可以使用以下命令检查 CPU 是否支持 AVX 等指令集，这对于某些模型的性能至关重要。
    ```shell
    cat /proc/cpuinfo | grep flags | head -1
    ```

### 7.2 贡献指南

Ollama 是一个开源项目，欢迎社区贡献。在提交代码时，请遵循良好的提交消息格式，例如：

*   **Good:** `llm/backend/mlx: support the llama architecture`
*   **Bad:** `feat: add more emoji`

### 7.3 相关链接

*   **官方网站:** [https://ollama.com/](https://ollama.com/)
*   **GitHub 仓库:** [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
*   **模型库:** [https://ollama.com/library](https://ollama.com/library)