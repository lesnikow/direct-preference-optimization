# TODOs


## Eval and graph of [1000, ... , 64000] train run prompt-matched

- [ ] Update next experiment on results here.

- [ ] Generate plot of series training.


- [ ] Batch conversion of models in `3_eval`.

- [ ] New `3_eval.py` translation of `3_eval.sh`.


## Train run of [1000, ... , 64000] prompt-matched

- [ ] Finish this training run.

- [x] Update training logic to do series training to have trains loaded up

  - [x] One of the following:

    - [x] Python script to handle this series training
    
      - [x] [Implementation of `2_dpo` in Python]

    ~~- [ ] Modify `2_dpo` to accept positional arg and do bash script `2_dpo_series`
          or `loop_2_dpo` or just a CLI for loop.~~


## Dataset sizes as in DPO paper

- [ ] Dataset sizes from 2 K to 64 K, randomly selected subreddits

  - [x] Prompt matched, so same exact prompts.

  - [ ] Size matched. One implementation: same exact prompts, then more prompts for `sc`.

- [x] Randomly selected topics, dataset sizes 300, 2.4 K.


## Size-matched dataset generation

- [ ] Update codebase to do size-matched dataset series generation,
      using top-up of sc approach.


## Metrics reporting

- [x] Metrics reporting to wandb in fastchat evals.


## Evals

- [ ] See why current evals with `no_train` gives lots of comparisons on BRC last run.
      Something going on where `no_train` lots of model answers or model judgements?


## Robustness

- [ ] Robustness of past positive results around `sc` 20K, both for `maj` 20k and `maj` topic
  matched ~40k.

- [ ] Robustness of results, seed integration into model training, answer generation, [model
  judgements].


## Ease of use

- [ ] Chain together `1_sft` and `2_dpo` scripts.
