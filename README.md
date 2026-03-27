How to run evaluation?

- Run `python3 -m evaluate.preparation all --dataset-name mmlu-pro`
- Run `python3 -m evaluate.preparation all --dataset-name MATH`
- Run `python3 -m evaluate.preparation all --dataset-name flores`
- Run `export IFBENCH_EVAL_ROOT=~/projects/IFBench` and then `python3 -m evaluate.preparation all --dataset-name mmlu-pro`
- Run `export OPENAI_CLIENT_CONFIG_PATH=~/projects/master-thesis/openai_configs.yaml`