# TODOs


## Eval and graph of [2000 ... 20000] train run

- [ ] Update next experiment on results here.
- [ ] Generate plot of series training.


## Train run of [2000 ... 20000] topic-matched

- [ ] Update training logic to do series training to have trains loaded up
  - [ ] One of the following:
    - [ ] Python script to handle this series training
      - [ ] Implementation of `3_dpo` in Python? 
    - [ ] Modify `3_dpo` to accept positional arg and do bash script `3_dpo_series`
          or `loop_3_dpo` or just a CLI for loop.


## Dataset sizes as in DPO paper

- [ ] Dataset sizes from 2.4 K to 20K, randomly selected subreddits, 
    - [ ] Prompt matched, so same exact prompts
    - [ ] Size matched. One implementation: same exact prompts, then more prompts for
      `sc`.

- [x] Randomly selected topics, dataset sizes 300, 2.4 K.


## Size-matched dataset generation

- [ ] Update code to do size-matched dataset series generation,
      using top-up of sc approach.


## Metrics reporting
- [x] Metrics reporting to wandb in fastchat evals


## Evals

- [ ] See why current evals with `no_train` gives lots of comparisons.


## Robustness

- [ ] Robustness of past positive results around `sc` 20K, both for `maj` 20k and `maj` topic
  matched ~40k.

- [ ] Robustness of results, seed integration into model training, answer generation, [model
  judgements].


## Ease of use

- [ ] Chain together `1_sft` and `2_dpo` scripts.
