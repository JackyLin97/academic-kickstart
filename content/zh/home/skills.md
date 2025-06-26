+++
# 使用Featurette小部件创建的技能部分。
widget = "featurette"  # 参见 https://sourcethemes.com/academic/docs/page-builder/
headless = true  # 这个文件代表一个页面部分。
active = false  # 激活这个小部件？true/false
weight = 30  # 此部分将出现的顺序。

title = "技能"
subtitle = ""

# 展示个人技能或业务特点。
#
# 根据需要添加/删除任意数量的`[[feature]]`块。
#
# 可用图标，请参见：https://sourcethemes.com/academic/docs/widgets/#icons

[[feature]]
  icon = "python"
  icon_pack = "fab"
  name = "Python"
  description = "PyTorch, TensorFlow, JAX"
  
[[feature]]
  icon = "brain"
  icon_pack = "fas"
  name = "大型语言模型"
  description = "LangChain, DSPy, LlamaIndex, PEFT"
  
[[feature]]
  icon = "database"
  icon_pack = "fas"
  name = "向量数据库"
  description = "Milvus, Weaviate, FAISS"

[[feature]]
  icon = "microchip"
  icon_pack = "fas"
  name = "模型部署"
  description = "vLLM, DeepSpeed, llama.cpp, ollama"

[[feature]]
  icon = "comments"
  icon_pack = "fas"
  name = "语音处理"
  description = "SpeechBrain, Whisper, ESPnet"

[[feature]]
  icon = "code"
  icon_pack = "fas"
  name = "开发工具"
  description = "C++, JavaScript, Docker, Git"

+++
