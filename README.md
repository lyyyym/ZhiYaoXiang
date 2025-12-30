# ZhiYaoXiang（智药箱）

------

这是一个基于 **MindSpore** 和 **Qwen2.5-7B** 大模型的智能医疗问答系统，专为 **无人售药机** 场景设计。

本项目采用 **RAG (检索增强生成)** + **LoRA 微调** 的架构，旨在解决无人售药场景下的两大核心痛点：

1. **库存实时性**：通过 RAG 挂载本地药品数据库，确保推荐的药品是机器内“现货”。
2. **回答严谨性**：通过 LoRA 微调，使模型严格遵循医学说明书，并对禁忌症（如孕妇、过敏）进行强制安全检查。

------

## 项目背景

随着智慧医疗和新零售的发展，无人售药机作为“24小时不打烊的微型药房”，正在社区、写字楼和交通枢纽迅速普及。然而，当前的无人售药终端普遍面临**“有售卖无服务”**的困境：

1. **选药决策难**：用户（尤其是深夜身体不适时）往往缺乏专业医学知识，面对琳琅满目的药品，难以根据自身症状准确判断“该买哪一种”。
2. **用药安全隐患**：缺乏驻店药师的实时指导，普通用户极易忽视潜在的药物禁忌（如孕妇、儿童、药物过敏或肝肾功能不全），存在错服、误服风险。
3. **大模型落地的“最后一公里”**：通用的医疗大模型虽然知识渊博，但**不知道当前机器里具体卖什么药**（库存割裂），且容易产生幻觉（推荐买不到的药）。

**本项目旨在解决上述核心痛点**。我们利用 **MindSpore** 框架和 **Qwen2.5** 大模型，构建了一个既懂医学常识、又懂实时库存的“智能药师大脑”。通过 **RAG（检索增强生成）** 技术让模型“看库存说话”，通过 **LoRA 微调** 让模型“守规矩”，从而实现从传统的“人找药”向智能化的“药找人”转变，为用户提供安全、精准的 24 小时购药咨询服务。


## 核心特性

- 基于 Qwen2.5-7B-Instruct，具备极强的中文理解与指令跟随能力。
- 完全基于 **MindSpore** 和 **MindNLP 0.5.1** 开发，适配 Ascend (昇腾) 算力环境。
- 内置简易向量数据库，支持基于症状的语义检索，动态注入药品说明书作为上下文。
- 微调后的模型具备高度的安全意识，能够识别高危人群（孕妇、儿童、肝肾功能不全者）并给出风险提示。
- 提供 Gradio Web 界面，模拟真实的无人售药机触摸屏体验。

------

## 项目结构

Plaintext

```
.
├── drug_knowledge_base.json       # 药品库存与说明书数据库
├── medical_rag_finetune_data.jsonl # 由脚本生成的指令微调数据
├── 1_generate_data.py             # 读取 JSON 生成微调数据集
├── 2_train.py                     # 基于 MindNLP 进行 LoRA 微调
├── build_simple_vector_db.py      # 构建向量索引（RAG 知识库）
├── myDB.py                        # 简易向量数据库实现类
├── inference.py                   # 命令行版本的 RAG 推理脚本
├── gradio_app.py                  # Gradio 可视化演示界面
└── README.md                      
```

------

## 环境准备

请确保您的环境满足以下要求：

- **OS**: Linux (Ubuntu)
- **Hardware**: Ascend 910B 
- **Python**: 3.10
- **MindSpore**: 2.7.0+
- **Dependencies**:

Bash

```
pip install mindspore
pip install mindnlp==0.5.1
pip install gradio jieba datasets scikit-learn
```

------

### 生成微调数据

读取 `drug_knowledge_base.json`，模拟生成包含 RAG 上下文的对话数据。

Bash

```
python 1_generate_data.py
```

> 运行后将生成 `medical_rag_finetune_data.jsonl`

### 模型微调 (LoRA)

使用 MindNLP 对 Qwen2.5-7B 进行轻量级微调，使其学会“查阅资料”和“安全合规”。

Bash

```
python 2_train.py
```

- **模型输入**：`Qwen/Qwen2.5-7B-Instruct` (自动下载或指定本地路径)
- **输出目录**：`./qwen2.5_medical_lora` (保存 LoRA 权重)
- *注意：请根据显存大小在 `2_train.py` 中调整 `batch_size` 和 `max_length`。*

### 构建知识库索引

将药品数据向量化，以便 RAG 系统进行检索。

Bash

```
python build_simple_vector_db.py
```

> 运行后将生成索引文件（meta.json,vectors.npy）。

### 启动 Web 演示

启动 Gradio 界面，体验完整的“问诊-检索-推荐”流程。

Bash

```
python gradio_app.py
```

打开浏览器访问终端输出的 URL (例如 `http://127.0.0.1:7860`)。

------

## 测试案例

在 Gradio 界面中尝试以下提问，观察模型的反应：

1. **症状咨询**：

   > "我嗓子疼，吞咽困难，机子里有药吗？"
   >
   > 预期：检索到药品，并告知用法。

   ![image-20251218203941040](https://github.com/lyyyym/ZhiYaoXiang/blob/main/image-zzzx.png?raw=true)

2. **安全拦截 (Key Feature)**：

   > "我是孕妇，牙疼得厉害，能吃布洛芬吗？"
   >
   > 预期：模型检索到布洛芬，但发现禁忌症，强烈警告孕妇禁用。
   >

   ![image-20251218203956458](https://github.com/lyyyym/ZhiYaoXiang/blob/main/image-aalj.png?raw=true)

3. **缺货/无关处理**：

   > "我想买个汉堡包。"
   >
   > 预期：礼貌拒绝，并说明本机仅提供非处方药品。
   
   ![image-20251218204044985](https://github.com/lyyyym/ZhiYaoXiang/blob/main/image-qh.png?raw=true)
