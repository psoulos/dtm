# Differtiable Tree Machine
This is the official repo accompanying the ICML 2023 publication [*Differentiable Tree Operations Promote Compositional
Generalization*](https://arxiv.org/abs/2306.00751).

## Setup
This repo has been tested with Python 3.8.13. You can install the necessary packages by running:
`pip install -r requirements.txt`. Some packages such as `pytorch` may need to be installed
via their own directions in order to install the correct version for your hardware (CPU or GPU).

## Data
Data available at https://huggingface.co/datasets/rfernand/basic_sentence_transforms. The primary four tasks used in the 
paper are: 

1. [car_cdr_seq](https://huggingface.co/datasets/rfernand/basic_sentence_transforms/resolve/main/car_cdr_seq.zip)
2. [active_logical_ttb](https://huggingface.co/datasets/rfernand/basic_sentence_transforms/resolve/main/active_logical_ttb.zip)
3. [passive_logical_ttb](https://huggingface.co/datasets/rfernand/basic_sentence_transforms/resolve/main/passive_logical_ttb.zip)
4. [actpass_logical_tt](https://huggingface.co/datasets/rfernand/basic_sentence_transforms/resolve/main/actpass_logical_tt.zip)

Each of these tasks should have a separate directory in `./data_files/`. For example, if you download
active_logical_ttb.zip, place the zip file in `./data_files/` and unzip it.
This should create the directory `data_files/active_logical_ttb/` which contains json files for the various data splits 
like train.jsonl, test.jsonl, etc. I recommend using [car_cdr_rcons.zip](https://huggingface.co/datasets/rfernand/basic_sentence_transforms/blob/main/car_cdr_rcons.zip) as a small task to get the code up and running.



## Primary Results
The following command can be used to train and evaluate the model used for the results in Table 1.

`python main.py --task_type=active_logical_ttb --dtm_steps=16 --max_tree_depth=12 --lr=1e-4 --steps=20000 --ctrl_hidden_dim=64
--transformer_norm_first=1 --wd=1e-1 --num_warmup_steps=10000 --gclip=1 --batch_size=16 --optim_beta2=.95 --train_log_freq=20 
--transformer_nheads=4 --scheduler=cosine --router_dropout=.1`

This command trains a model on the active↔logical task. Change `--task_type` to one of the other tasks to train and evaluate
models for that specific task. All of the hyperparameters remain the same for each task, except for `--dtm_steps`. This should
be set according to each task:

```
car_cdr_seq: 12 steps
active_logical_ttb: 16 steps
passive_logical_ttb: 28 steps
actpass_logical_tt: 20 steps
```
## Learned Structural Transformation Ablation
To generate the values shown in Table 2, you can have the E and D matrices be learned by using the commandline argument
`--predefined_operations_are_random`.

## Gumbel Softmax ablations
To generate the values shown in Table 3, you can turn on Gumbel Softmax for argument selection by using the commandline
argument `--arg_dist_fn=gumbel`. Similarly, you can turn on Gumbel Softmax for operation selection by using the 
commandline argument `--op_dist_fn=gumbel`.



## wandb
You can set the $WANDB_API_KEY environment variable to use wandb.

## Citation
If you use this code, please cite the work:
```bibtex
@InProceedings{pmlr-v202-soulos23a,
  title = 	 {Differentiable Tree Operations Promote Compositional Generalization},
  author =       {Soulos, Paul and Hu, Edward J and Mccurdy, Kate and Chen, Yunmo and Fernandez, Roland and Smolensky, Paul and Gao, Jianfeng},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {32499--32520},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/soulos23a/soulos23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/soulos23a.html},
  abstract = 	 {In the context of structure-to-structure transformation tasks, learning sequences of discrete symbolic operations poses significant challenges due to their non-differentiability. To facilitate the learning of these symbolic sequences, we introduce a differentiable tree interpreter that compiles high-level symbolic tree operations into subsymbolic matrix operations on tensors. We present a novel Differentiable Tree Machine (DTM) architecture that integrates our interpreter with an external memory and an agent that learns to sequentially select tree operations to execute the target transformation in an end-to-end manner. With respect to out-of-distribution compositional generalization on synthetic semantic parsing and language generation tasks, DTM achieves 100% while existing baselines such as Transformer, Tree Transformer, LSTM, and Tree2Tree LSTM achieve less than 30%. DTM remains highly interpretable in addition to its perfect performance.}
}
```
