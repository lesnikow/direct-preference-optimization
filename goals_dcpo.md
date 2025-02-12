



# Reference implementation

+ SFT, DPO, eval model on SHP from reference implementation
  + Eval DPO 
  + Eval SFT  
  + DPO using default repo behavior
  + SFT using default repo behavior


# Updated data format DPO and evals

+ DPO and eval models on improved data format
  + Eval DPO 
  + Train DPO

+ Regen DCPO training data from notebook with ...
  + Updated prompts based on how fastchat evaluates pythia models
  + Updated json dumps methods, to fix utf-8 error, and encode ' correctly.

+ Update notebook
  + Make method for bottom write out method
  + One format for prompt method, refactor


# SFT training
    
+ D: Update of converted model save in dpo repo during train
  + D: Test
  + D: Be ready to undo specific commit that made this change to model save method, 
    as needed to get sft training rolling.

+ D: Try training on SFT data again 
  + D: BRC try py 1-sft.py
  + D: BRC pull
  + D: Commit updated SFT data

+ D: Regen sft condorcet training data from notebook with ...
  + D: Get rejected: None for sft data
  + D: [Updated prompts based on how fastchat evaluates pythia models]
  + D: Updated json dumps methods, to fix utf-8 error, and encode ' correctly.
