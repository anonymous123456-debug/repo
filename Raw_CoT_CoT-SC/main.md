# Raw / CoT / CoT-SC Baseline Description

This directory contains baseline experiment code for three reasoning paradigms: Raw, CoT, and CoT-SC.

## Three Paradigms

- `raw`: Direct answer generation without explicitly requiring the model to output a step-by-step reasoning process.
- `cot`: A single Chain-of-Thought reasoning path, where the model derives the answer step by step.
- `cotsc`: Chain-of-Thought Self-Consistency, where multiple reasoning chains are generated and the answer is selected by voting (5 chains).

## File Contents

- `2wiki.py`: Raw / CoT / CoT-SC experiments for 2WikiMultiHopQA.
- `commonsense.py`: Raw / CoT / CoT-SC experiments for CommonsenseQA.
- `cosmos.py`: Raw / CoT / CoT-SC experiments for CosmosQA.
- `hotpotqa.py`: Raw / CoT / CoT-SC experiments for HotPotQA.
- `logicbench.py`: Raw / CoT / CoT-SC experiments for LogicBench BQA and MCQA.
- `medqa.py`: Raw / CoT / CoT-SC experiments for MedQA.
- `sciq.py`: Raw / CoT / CoT-SC experiments for SciQ.
- `squad.py`: Raw / CoT / CoT-SC experiments for SQuAD.
- `winograd.py`: Raw / CoT / CoT-SC experiments for Winograd pronoun resolution tasks.
- `entropy.py`: Attention entropy analysis code.
- `metric.py`: Metric calculation tools for Exact Match, F1, BLEU, ROUGE, and related scores.
- `sh/run_experiment.sh`: Execution script.

## Execution Process

1. Prepare the model. Do not hardcode personal model paths in the code; specify the model with `--model_path` or the `MODEL_PATH` environment variable.
2. Prepare the data. Each script reads data from the relative `./data/...` path defined in that script.
3. Select the dataset script and method: `raw`, `cot`, or `cotsc`.
4. Run the script. Results will be written to the corresponding directory or file under `./analysis/...`.

Generic script examples:

```bash
bash sh/run_experiment.sh hotpotqa raw <MODEL_PATH>
bash sh/run_experiment.sh hotpotqa cot <MODEL_PATH>
bash sh/run_experiment.sh hotpotqa cotsc <MODEL_PATH>
```

Direct Python examples:

```bash
python hotpotqa.py --method raw --model_path <MODEL_PATH>
python commonsense.py --dataset commonsense --method cotsc --model_path <MODEL_PATH>
```
