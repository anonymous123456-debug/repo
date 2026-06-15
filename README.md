# Are Reasoning Paradigms Scale-Aware? A Cross-Paradigm Verification of Prompting, Retrieval, and Knowledge-Graph Scaffolding for Small Language Models

## Abstract

In recent years, reasoning-enhancement paradigms such as chain-based prompting, retrieval-augmented generation, and graph-guided reasoning have significantly improved the performance of large language models on question answering, multi-hop reasoning, and knowledge-intensive tasks. However, whether these methods can be stably transferred to small language models remains insufficiently verified. In real-world deployment, small language models offer advantages such as low cost, low latency, local deployability, and stronger privacy control, but they are also more vulnerable to prompt complexity, retrieval noise, context length, and the organization of external evidence. Therefore, evaluating the effectiveness of different reasoning-enhancement paradigms on smaller models is not only a lightweight deployment issue, but also an important entry point for understanding the boundary conditions of reasoning paradigms.

This repository provides the code implementation and reproducibility package for the question of whether small language models can reliably benefit from different reasoning-enhancement paradigms. The experiments cover three major method families: prompting-based reasoning, text-retrieval-based augmentation, and knowledge-graph-based structured scaffolding. Under a unified experimental protocol, we compare representative paradigms including CoT, CoT-SC, ToT, RAG, LongRAG, GraphRAG, MindMap, and ToG, and analyze their behavioral differences in small-model settings from the perspectives of accuracy, efficiency, robustness, evidence organization, inference latency, and Token cost.

## Our Contributions

The main contributions stated in the paper are as follows:

1. **Scale-aware transfer analysis.** This study examines whether reasoning-enhancement paradigms originally designed for large language models, including chain-based prompting, retrieval augmentation, and graph-guided scaffolding, remain effective when transferred to 1.7B-8B small language models.
2. **Evidence-interface perspective.** This study frames reasoning enhancement as an interaction between model capacity and evidence organization, and compares prompting, unstructured textual evidence, and structured relational evidence as support interfaces for small language models.
3. **Unified E3 evaluation framework.** This study evaluates representative methods under a shared experimental protocol from three dimensions: effectiveness, efficiency, and robustness. The evaluation covers not only accuracy, but also Token usage, inference latency, performance stability, and small-model compatibility.
4. **Empirical boundary findings.** The experimental results show that reasoning enhancement is not scale-neutral: chain-based prompting can be unstable on ultra-small models, unstructured retrieval often faces cost-effectiveness limitations, while structured relational evidence shows more stable gains in multi-hop and knowledge-intensive tasks.

## Framework

![Framework](assets/framework.png)

## Repository Overview

This code repository contains the following materials:

1. dataset processing scripts, dataset splits, and sampling lists;
2. all prompt templates and answer-normalization decision rules;
3. retrieval model and text chunking configuration parameters;
4. graph construction scripts and knowledge graph statistics;
5. algorithm-specific configuration files, including CoT, CoT-SC, ToT, RAG, LongRAG, GraphRAG, MindMap, and ToG;
6. run-log scripts or statistic locations for recording inference latency and Token consumption;
7. evaluation code and statistical analysis code.

## Baselines

Each baseline folder contains its own implementation code, execution scripts, and detailed documentation:

| Folder | Baseline |
| --- | --- |
| `Raw_CoT_CoT-SC/` | Raw prompting, Chain-of-Thought, and Self-Consistency CoT |
| `ToT/` | Tree of Thoughts |
| `LongRAG_and_RAG/` | RAG and LongRAG |
| `graphrag/` | GraphRAG |
| `mindmap/` | MindMap and knowledge graph construction |
| `ToG/` | Think-on-Graph |

The execution scripts and specific configuration information for each baseline are described in detail in the corresponding `main.md` or documentation file. The knowledge graph construction process is described in the MindMap-related documentation. RAG corpus preparation and retrieval index construction are described in `LongRAG_and_RAG/main.md`.

## Reproducibility Notes

Answer decision rules, evidence extraction, answer normalization, and evaluation procedures are all included in the corresponding baseline implementations. Inference latency, Token consumption, intermediate outputs, and final statistical results should be checked through runtime printed logs or final log files.

For reproduction, please carefully read the `main.md` of each method and follow the corresponding steps and explanations.

This repository does not include large model weights or private runtime resources. Please download the required embedding, reranking, and generation models yourself according to the instructions in each baseline folder.

## Knowledge Graph Size Statistics

The following table shows the size of the knowledge graphs constructed for each dataset. The bar charts are normalized by the maximum value in this table, making it easier to quickly compare the number of entity nodes and relationships across datasets.

| Dataset | #Entity Nodes | Node Scale | #Relationships | Relationship Scale |
| --- | ---: | --- | ---: | --- |
| 2WikiMultiHopQA | 575 | ██████░░░░ | 1492 | ██████░░░░ |
| LogicBench-BQA | 166 | ██░░░░░░░░ | 538 | ██░░░░░░░░ |
| CommonsenseQA | 355 | ████░░░░░░ | 711 | ███░░░░░░░ |
| CosmosQA | 325 | ███░░░░░░░ | 1012 | ████░░░░░░ |
| HotpotQA | 977 | ██████████ | 2359 | █████████░ |
| LogicBench-MCQA | 78 | █░░░░░░░░░ | 140 | █░░░░░░░░░ |
| MedQA | 711 | ███████░░░ | 2504 | ██████████ |
| SciQ | 36 | █░░░░░░░░░ | 70 | █░░░░░░░░░ |
| SQuAD | 129 | █░░░░░░░░░ | 264 | █░░░░░░░░░ |
| Winograd | 91 | █░░░░░░░░░ | 285 | █░░░░░░░░░ |

## Citation

If you use this codebase, please cite the corresponding paper once the final citation information is available.
