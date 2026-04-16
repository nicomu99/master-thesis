How to run inference?

- Run `python3 -m inference.app --path ...` first and create personas

How to run evaluation?

- Run `python3 -m evaluate.preparation all --dataset-name mmlu-pro`
- Run `python3 -m evaluate.preparation all --dataset-name MATH`
- Run `python3 -m evaluate.preparation all --dataset-name flores`
- Run `python3 -m evaluate.preparation all --dataset-name alpaca`
- Run `export IFBENCH_EVAL_ROOT=~/projects/IFBench` and then `python3 -m evaluate.preparation all --dataset-name mmlu-pro`