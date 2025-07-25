---
title: "大型语言模型超参数调优指南：从生成到部署的全面解析"
subtitle: ""
summary: "本文深入解析了大型语言模型(LLM)的两大类关键超参数：生成超参数和部署超参数，详细阐述了它们的作用、取值范围、影响以及在不同场景下的最佳实践，帮助开发者精确调整模型以获得理想的性能、成本和输出质量。"
authors:
- admin
language: zh
tags:
- 大型语言模型
- 超参数调优
- 模型部署
- API调用
- vLLM
categories:
- 技术文档
- 机器学习
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

## 引言

<span style="font-size: 0.9em;">大型语言模型（LLM）的强大能力背后，是一系列复杂的超参数在"默默奉献"。无论是在本地部署一个像 vLLM 一样的推理服务，还是调用 OpenAI 的 API，精确地调整这些参数对于获得理想的性能、成本和输出质量至关重要。这份文档将"掰开了，揉碎了"地深入解析两大类关键超参数：**生成（Sampling）超参数** 和 **部署（Serving）超参数**，帮助你完全掌握它们的作用、取值、影响以及在不同场景下的最佳实践。</span>
---

### 第一部分：生成（Sampling）超参数——掌控模型的创造力与确定性

生成超参数直接控制模型在生成下一个 token 时的行为。它们主要围绕着一个核心问题：如何在模型给出的成千上万个可能的下一个词的概率分布中进行选择。

### 1. `temperature` (温度)

**一句话解释：** 控制生成文本的随机性。`temperature` 越高，随机性越强，回答越具创造性和多样性；`temperature` 越低，随机性越弱，回答越趋于确定性和保守。

*   **底层原理：**
    在生成下一个 token 时，模型会为词汇表中的所有词计算一个 `logits`（原始的、未归一化的预测分数）。通常，我们会使用 `Softmax` 函数将这些 `logits` 转换成一个概率分布。`temperature` 参数在 `Softmax` 计算之前被引入，它会"平滑"或"锐化"这个概率分布。

    标准的 Softmax 公式是： `P(i) = exp(logit_i) / Σ_j(exp(logit_j))`

    引入 `temperature` (T) 后的公式是：`P(i) = exp(logit_i / T) / Σ_j(exp(logit_j / T))`

    *   当 `T` -> 0 时，`logit_i / T` 的差异会急剧拉大。拥有最高 logit 的那个 token 的概率会无限接近 1，而其他所有 token 的概率会无限接近 0。这使得模型几乎总是选择最有可能的那个词，表现得非常确定和"贪心"。
    *   当 `T` = 1 时，公式回归标准 Softmax，模型的行为就是其"原始"状态。
    *   当 `T` > 1 时，`logit_i / T` 的差异会被缩小。原本概率较低的 token 的概率会被提升，整个概率分布变得更加"平坦"。这增加了模型选择到不那么常见的词的几率，从而引入了更多的随机性和创造性。

*   **取值范围与建议：**
    *   **范围:** `[0.0, 2.0]` (理论上可以更高, 但 OpenAI API 通常限制在 2.0)。
    *   **`temperature` = 0.0:** 适用于需要确定性、可复现和高准确度输出的场景。例如：代码生成、事实问答、文本分类、数据提取。每次输入相同，输出也几乎完全相同（除非模型本身有更新）。
    *   **低 `temperature` (例如 `0.1` - `0.4`):** 适用于需要严谨、忠于原文的半创作性任务。例如：文章摘要、翻译、客服机器人。输出会略有变化，但大体上忠实于核心内容。
    *   **中等 `temperature` (例如 `0.5` - `0.8`):** 创造性与一致性的良好平衡点，是大多数应用场景的默认和推荐值。例如：撰写邮件、市场文案、头脑风暴。
    *   **高 `temperature` (例如 `0.9` - `1.5`):** 适用于高度创造性的任务。例如：写诗、创作故事、生成对话脚本。输出会非常多样，甚至可能出人意料，但有时也可能产生无意义或不连贯的内容。

*   **注意事项:**
    *   `temperature` 和 `top_p` 通常不建议同时修改，最好只调整其中一个。OpenAI 的文档也明确指出，通常建议只修改其中之一。

### 2. `top_p` (核心采样)

**一句话解释：** 通过保留一个累积概率阈值（`p`）内的最高概率词汇，来动态地决定采样池的大小，从而控制生成的多样性。

*   **底层原理：**
    `top_p` 是一种比 `temperature` 更智能的采样策略，也称为 **核心采样 (Nucleus Sampling)**。它不是调整所有 token 的概率，而是直接划定一个"核心"候选集。

    具体步骤如下：
    1.  模型计算出所有候选 token 的概率分布。
    2.  将所有 token 按概率从高到低排序。
    3.  从概率最高的 token 开始，依次累加它们的概率，直到这个累积概率总和超过设定的 `top_p` 阈值。
    4.  所有被累加过的这些 token 构成了采样的"核心集合"（nucleus）。
    5.  模型将只从这个核心集合中进行采样（通常会重新归一化它们的概率），所有其他 token 将被忽略。

    **举个例子：** 假设 `top_p` = `0.9`。
    *   如果概率最高的 token "the" 的概率是 `0.95`，那么核心集合里就只有 "the" 这一个词，模型会 100% 选择它。
    *   如果 "the" 的概率是 `0.5`，"a" 的概率是 `0.3`，"an" 的概率是 `0.1`，那么这三个词的累积概率是 `0.9`。核心集合就包含 {"the", "a", "an"}。模型将从这三个词中按其（重新归一化的）概率进行采样。

*   **取值范围与建议：**
    *   **范围:** `(0.0, 1.0]`。
    *   **`top_p` = 1.0:** 意味着模型会考虑所有 token，不进行任何截断（等同于没有 `top_p`）。
    *   **高 `top_p` (例如 `0.9` - `1.0`):** 允许更多样化的选择，适用于创造性任务，效果类似于较高的 `temperature`。
    *   **低 `top_p` (例如 `0.1` - `0.3`):** 极大地限制了模型的选择范围，使其输出非常确定和保守，效果类似于极低的 `temperature`。
    *   **通用建议值:** `0.9` 是一个非常常见的默认值，因为它在保持高质量的同时，也允许一定的多样性。

*   **`top_p` vs `temperature`:**
    *   `top_p` 更加动态和自适应。在模型对下一步非常确信时（概率分布很尖锐），`top_p` 会自动缩小候选集，保证质量。在模型不那么确信时（概率分布很平坦），它会扩大候选集，增加多样性。
    *   `temperature` 则是"一视同仁"地调整整个分布，不管分布本身是尖锐还是平坦。
    *   因此，`top_p` 通常被认为是比 `temperature` 更安全、更鲁棒的控制多样性的方法。

### 3. `top_k`

**一句话解释：** 简单粗暴地只从概率最高的 `k` 个 token 中进行采样。

*   **底层原理：** 这是最简单的截断采样方法。直接选择概率最高的 `k` 个 token，组成候选集，然后从这 `k` 个 token 中进行采样。所有其他 token 都被忽略。

*   **取值范围与建议：**
    *   **范围:** 整数，例如 `1`, `10`, `50`。
    *   **`top_k` = 1:** 等同于贪心搜索，总是选择最有可能的词。
    *   **建议:** `top_k` 通常不作为首选的采样策略，因为它太"死板"。在某些概率分布非常平坦的情况下，它可能会意外地排除掉很多合理的词；而在分布非常尖锐时，它又可能包含进很多概率极低的无用词。`top_p` 通常是更好的选择。

### 4. `repetition_penalty` (重复惩罚)

**一句话解释：** 对在上下文中已经出现过的 token 施加惩罚，以降低它们再次被选中的概率，从而减少重复内容。

*   **底层原理：** 在计算 `logits` 后，但在 `Softmax` 之前，该参数会遍历所有候选 token。如果一个 token 已经在之前的上下文中出现过，它的 `logit` 值就会被降低（通常是除以 `repetition_penalty` 的值）。

    `new_logit = logit / penalty` (如果 token 已出现)
    `new_logit = logit` (如果 token 未出现)

    这样，已经出现过的词的最终概率就会下降。

*   **取值范围与建议：**
    *   **范围:** `1.0` 到 `2.0` 之间比较常见。
    *   **`1.0`:** 不施加任何惩罚 (默认值)。
    *   **`1.1` - `1.3`:** 是一个比较安全的范围，可以有效减少不必要的重复，而不过度影响正常的语言表达（比如必要的冠词 "the"）。
    *   **过高的值:** 可能会导致模型刻意回避常用词，产生不自然甚至奇怪的句子。
### 5. `frequency_penalty` & `presence_penalty` (频率与存在感惩罚)

这两个参数是 `repetition_penalty` 的更精细化版本。

*   **`presence_penalty` (存在感惩罚):**
    *   **作用:** 对所有在上下文中 **至少出现过一次** 的 token 施加一个固定的惩罚。它不关心这个 token 出现了多少次，只要出现过，就惩罚。
    *   **底层原理:** `new_logit = logit - presence_penalty` (如果 token 至少出现过一次)。
    *   **场景:** 当你想鼓励模型引入全新的概念和词汇，而不是反复讨论已经提到过的话题时，这个参数很有用。
    *   **范围:** `0.0` 到 `2.0`。正值会惩罚新 token，负值会鼓励。

*   **`frequency_penalty` (频率惩罚):**
    *   **作用:** 惩罚的大小与 token 在上下文中出现的 **频率** 成正比。一个词出现的次数越多，它受到的惩罚就越重。
    *   **底层原理:** `new_logit = logit - count(token) * frequency_penalty`。
    *   **场景:** 当你发现模型倾向于反复使用某些特定的高频词（即使它们是必要的），导致语言单调时，这个参数可以有效降低这些词的概率。
    *   **范围:** `0.0` 到 `2.0`。

*   **总结:** `presence_penalty` 解决"是否出现过"的问题，`frequency_penalty` 解决"出现了多少次"的问题。

### 6. `seed` (随机种子)

**一句话解释：** 通过提供一个固定的 `seed`，可以使得在其他参数（如 `temperature`）相同的情况下，模型的输出是可复现的。

*   **作用:** 在机器学习中，很多操作看似随机，实则是"伪随机"，它们由一个初始的"种子"决定。设置相同的种子，就能得到相同的随机数序列。在 LLM 中，这意味着采样过程将是完全确定的。
*   **场景:**
    *   **调试与测试:** 当你需要验证某个改动是否影响了输出时，固定 `seed` 可以排除随机性干扰。
    *   **可复现的研究:** 在学术研究中，可复现性至关重要。
    *   **生成一致性内容:** 当你需要模型对同一输入始终产生相同风格的输出时。
*   **注意:** 要想完全复现，**所有** 生成参数（`prompt`, `model`, `temperature`, `top_p` 等）都必须完全相同。

---

### 第二部分：部署（Serving）超参数——优化服务的性能与容量

部署超参数决定了 LLM 推理服务如何管理 GPU 资源、处理并发请求以及优化整体吞吐量和延迟。这些参数在 vLLM 这样的高性能推理引擎中尤为重要。

### 1. `gpu_memory_utilization`

**一句话解释：** 控制 vLLM 可以使用的 GPU 显存的比例，核心用途是为 **KV Cache** 预留空间。

*   **底层原理 (PagedAttention):**
    vLLM 的核心是 PagedAttention 机制。传统的注意力机制会为每个请求预分配一个连续的、最大长度的显存空间来存储 Key-Value (KV) Cache。这导致了严重的内存浪费，因为大部分请求的长度都远小于最大长度。

    PagedAttention 将 KV Cache 像操作系统的虚拟内存一样进行管理：
    1.  它将每个序列的 KV Cache 拆分成很多小的、固定大小的"块"（Block）。
    2.  这些块可以非连续地存储在 GPU 显存中。
    3.  一个中央的"块管理器"（Block Manager）负责分配和释放这些块。

    `gpu_memory_utilization` 正是告诉 vLLM："你可以用掉总显存的这么多比例来自由管理（主要是存放模型权重和 KV Cache 的物理块）"。

*   **取值范围与影响：**
    *   **范围:** `(0.0, 1.0]`。
    *   **默认值:** `0.9` (即 90%)。
    *   **值越高 (例如 `0.95`):**
        *   **优点:** vLLM 有更多的显存用于 KV Cache，可以支持更长的上下文、更大的批处理大小（batch size），从而提高吞吐量。
        *   **风险:** 如果设置得太高，可能会没有足够的备用显存留给 CUDA 内核、驱动或其他系统进程，容易导致 **OOM (Out of Memory)** 错误。
    *   **值越低 (例如 `0.8`):**
        *   **优点:** 更安全，不易 OOM，为系统和其他应用保留了更多显存。
        *   **缺点:** KV Cache 的可用空间变小，可能导致 vLLM 无法处理高并发或长序列请求，性能下降。当 KV Cache 不足时，vLLM 会触发 **抢占 (Preemption)**，将一些正在运行的序列换出，等待有足够空间后再换入，这会严重影响延迟。vLLM 的警告日志 `"there is not enough KV cache space. This can affect the end-to-end performance."` 就是在提醒你这一点。

*   **建议:**
    *   从默认值 `0.9` 开始。
    *   如果遇到 OOM，适当调低此值。
    *   如果遇到大量抢占警告，且确认没有其他进程占用大量显存，可以适当调高此值。

### 2. `max_num_seqs`

**一句话解释：** 限制 vLLM 调度器在 **一个迭代（或一个批处理）中** 可以处理的最大序列（请求）数量。

*   **底层原理:**
    vLLM 的调度器会在每个处理周期，从等待队列中选择一批请求来共同执行。这个参数直接限制了这个"批"的大小。它与 `max_num_batched_tokens`（限制一个批次中所有序列的总 token 数）共同决定了批处理的规模。

*   **取值范围与影响:**
    *   **范围:** 正整数，例如 `16`, `64`, `256`。
    *   **值越高:**
        *   **优点:** 允许更高的并发度，可能提高 GPU 的利用率和整体吞吐量。
        *   **缺点:** 需要更多的中间内存（例如，存储 `logits` 和采样状态），并可能增加单个批处理的延迟。如果设置得过高，即使 KV Cache 还有空间，也可能因为其他临时内存不足而 OOM。
    *   **值越低:**
        *   **优点:** 对内存更友好，单个批次延迟可能更低。
        *   **缺点:** 限制了并发能力，可能导致 GPU 利用率不足，吞吐量下降。

*   **建议:**
    *   这个值需要根据你的 GPU 显存大小、模型大小和预期的并发负载来调整。
    *   对于高并发场景，可以尝试逐步增加此值，并监控 GPU 利用率和内存使用情况。
    *   对于交互式、低延迟要求的场景，可以适当调低此值。
### 3. `max_model_len`

**一句话解释：** 设定模型能够处理的 **最大上下文长度**（包括 prompt 和生成的 token）。

*   **底层原理:**
    这个参数直接决定了 vLLM 需要为 KV Cache 预留多大的逻辑空间。例如，如果 `max_model_len` = `4096`，vLLM 就必须确保其内存管理机制能够支持每个序列最多存储 `4096` 个 token 的 KV 对。
    这会影响 vLLM 启动时的内存规划，比如 Position Embedding 的大小。

*   **取值范围与影响:**
    *   **范围:** 正整数，不能超过模型原始训练时的最大长度。
    *   **值越高:**
        *   **优点:** 可以处理更长的文档、更复杂的上下文。
        *   **缺点:** **显著增加** 内存消耗。每个 token 都需要存储 KV Cache，长度翻倍，内存占用也大致翻倍。即使当前请求很短，vLLM 也需要为潜在的长请求做好准备，这会占用更多的 KV Cache 块。
    *   **值越低:**
        *   **优点:** **显著节省** 显存。如果你知道你的应用场景永远不会超过 1024 个 token，那么将此值设为 1024 会比默认的 4096 或 8192 释放出大量的 KV Cache 空间，从而支持更高的并发。
        *   **缺点:** 任何超过此长度的请求都会被拒绝或截断。

*   **建议:**
    *   **按需设置！** 这是优化 vLLM 内存使用的最有效参数之一。根据你的实际应用场景，将此值设置为一个合理的、略带余量的最大值。

### 4. `tensor_parallel_size` (张量并行) & `pipeline_parallel_size` (流水线并行)

这两个参数用于在多个 GPU 或多个节点上部署超大模型。

*   **`tensor_parallel_size`:**
    *   **作用:** 将模型的 **每一层**（比如一个大的权重矩阵）都切分成 `N` 份（`N` = `tensor_parallel_size`），分别放到 `N` 个 GPU 上。在计算时，每个 GPU 只处理它自己那一部分的数据，然后通过高速互联（如 NVLink）交换必要的结果（All-Reduce 操作），最后合并得到完整输出。
    *   **场景:** 当单个模型的体积超过单张 GPU 的显存时使用。例如，一个 70B 的模型无法放入一张 40GB 的 A100，但可以设置 `tensor_parallel_size=2` 部署在两张 A100 上。
    *   **影响:**
        *   **优点:** 实现了模型并行，解决了单卡存不下的问题。
        *   **缺点:** 引入了大量的跨 GPU 通信开销，可能会影响延迟。需要 GPU 之间有高速互联。

*   **`pipeline_parallel_size`:**
    *   **作用:** 将模型的 **不同层** 分配到不同的 GPU 或节点上。例如，将 1-10 层放在 GPU 1，11-20 层放在 GPU 2，以此类推。数据像流水线一样流过这些 GPU。
    *   **场景:** 当模型非常非常大，需要跨多个节点（机器）部署时。
    *   **影响:**
        *   **优点:** 可以将模型扩展到任意数量的 GPU/节点。
        *   **缺点:** 会产生"流水线气泡"（pipeline bubble）的额外开销，即在流水线的开始和结束阶段，部分 GPU 会处于空闲等待状态，降低了利用率。

*   **组合使用:**
    vLLM 支持同时使用这两种并行策略，以在大型集群上高效部署巨型模型。

---

### 总结与最佳实践

| 场景 | `temperature` | `top_p` | `repetition_penalty` | `gpu_memory_utilization` | `max_num_seqs` | `max_model_len` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **代码生成/事实问答** | `0.0` - `0.2` | (不建议修改) | `1.0` | `0.9` (默认) | 根据并发调整 | 按需设置 |
| **文章摘要/翻译** | `0.2` - `0.5` | (不建议修改) | `1.1` | `0.9` | 根据并发调整 | 设为文档最大可能长度 |
| **通用聊天/文案写作**| `0.7` (默认) | `0.9` (推荐) | `1.1` - `1.2` | `0.9` | 根据并发调整 | 按需设置，例如`4096`|
| **创意写作/头脑风暴**| `0.8` - `1.2` | `0.95` | `1.0` | `0.9` | 根据并发调整 | 按需设置 |
| **高并发吞吐量优化**| (根据任务) | (根据任务) | (根据任务) | 尝试 `0.9` - `0.95` | 逐步调高 | 设为满足业务的**最小值** |
| **低延迟交互优化** | (根据任务) | (根据任务) | (根据任务) | `0.9` (默认) | 设为较低值 (如`16-64`)| 按需设置 |
| **内存极度受限** | (根据任务) | (根据任务) | (根据任务) | 调低至 `0.8` | 设为较低值 | 设为满足业务的**最小值** |

**最终建议：**
1.  **从生成参数开始调优：** 首先通过调整 `temperature` 或 `top_p` 获得满意的输出质量。
2.  **按需设置部署参数：** 在部署时，首先根据你的应用场景，将 `max_model_len` 设置为一个合理的最小值。
3.  **监控并迭代：** 使用默认的 `gpu_memory_utilization=0.9` 和一个适中的 `max_num_seqs` 开始。通过监控工具（如 `nvidia-smi` 和 vLLM 的日志）观察显存使用率和抢占情况，然后逐步迭代调整这些值，以在你的特定硬件和负载下找到最佳的平衡点。