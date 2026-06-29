# 数据预处理

本章节介绍全部数据集的预处理代码存放位置和RAG语料的预处理语料构建代码存放位置，并且给出脚本执行代码，代码位置位于/prepreocess目录下。我们还给出所有数据集的公开地址，我们实验所选取的数据子集存放于/mindmap/dataset下。

## 数据集来源

| 数据集 | 来源 |
| --- | --- |
| CommonsenseQA | https://hf-mirror.com/datasets/tau/commonsense_qa |
| CosmosQA | https://hf-mirror.com/datasets/allenai/cosmos_qa |
| Winograd | https://hf-mirror.com/datasets/marcov/winograd_wsc_wsc273_promptsource |
| SciQ | https://hf-mirror.com/datasets/allenai/sciq |
| MedQA | https://github.com/jind11/MedQA |
| HotpotQA | https://huggingface.co/datasets/hotpotqa/hotpot_qa |
| 2WikiMultiHopQA | https://huggingface.co/datasets/framolfese/2WikiMultihopQA |
| SQuAD | https://hf-mirror.com/datasets/rajpurkar/squad |
| LogicBench BQA | https://github.com/Mihir3009/LogicBench |
| LogicBench MCQA | https://github.com/Mihir3009/LogicBench |

## 统一参数

所有脚本均支持下列参数：

```bash
python <script>.py \
  --source_path "" \
  --output_dir outputs \
  --split train \
  --seed 931 \
  --sample_size 2000 \
  --eval_size 200
```

`--source_path` 默认留空。对于 Hugging Face 数据集，脚本会使用公开数据集名称加载；如果已经手动下载到本地，可以将 `--source_path` 指向本地数据集目录或文件。对于 MedQA 和 LogicBench，需要先从 GitHub 下载数据，再通过 `--source_path` 指向下载后的文件或目录。

## 执行示例

CommonsenseQA：

```bash
python commonsenseqa.py --output_dir outputs
```

SciQ：

```bash
python sciq.py --output_dir outputs
```

MedQA：

```bash
python medqa.py --source_path "" --output_dir outputs
```

LogicBench BQA：

```bash
python logicbench_bqa.py --source_path "" --output_dir outputs
```

LogicBench MCQA：

```bash
python logicbench_mcqa.py --source_path "" --output_dir outputs
```

## RAG 语料与索引构建

RAG/LongRAG 使用的语料生成、文本分块和向量索引构建流程已经整理到：

```text
/corpus_index
```

`corpus_index/` 中包含十个数据集的语料构建脚本，能够生成 `data/corpus/raw/<dataset>.json`、`chunks.json`、`id_to_rawid.json` 和 `vector.index`。具体字段抽取规则、默认分块参数和执行命令见：

```text
../corpus_index/main.md
```
