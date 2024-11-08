# TODOs

## Next experiment



## Eval and graph of [1000, ... , 64000] train run prompt-matched

- [x] Update next experiment on results here.

- [ ] Robustness of current plot, across diff random seeds
      Asked for in latest ICML / NeurIPS / ICLR instructions / checklist?

- [ ] Zero-point of training, e.g. no DPO steps, on graph

- [ ] Add baseline horizontal line to win-rate plot
      Possible baselines: pythia69b, 
                          pythia69b sft with maj, sc

- [P] Win-rate on y-axis, as in DPO paper 
      Possible baselines: pythia69b, 
                          pythia69b sft with maj, sc

- [x] Update plotting of sc series on x-axis, number of samples

- [x] Generate plot of series training.

- [x] Make model judgements in `3_eval.py`.

- [x] Make model answers in `3_eval.py`.

- [x] Batch conversion of models in `3_eval.py`.  


- [x] Debug min len key error in making model answers
  - [x] Triage if issue is in dpo or convert model step
    - [x] ~~Take a 1000 trained model, see if bash 3 eval script works step-by-step
        Result: Not same error when doing 3 eval sh script with 8000 trained model! 
          -> Try working on model conversion + answers in python, testing~~


- [x] Do fastchat setup and show results in new py module.

- [x] New `3_eval.py` translation of `3_eval.sh`.

- [x] ~~|Update eval py script to do python module imports instead of subprocesses|~~


## Train run of [1000, ... , 64000] prompt-matched

- [x] Finish this training run.

- [x] Update training logic to do series training to have trains loaded up

  - [x] One of the following:

    - [x] Python script to handle this series training
    
      - [x] [Implementation of `2_dpo` in Python]

    - [x] ~~Modify `2_dpo` to accept positional arg and do bash script `2_dpo_series`
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
