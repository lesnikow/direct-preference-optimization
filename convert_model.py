import torch

# Define the path to the input model and the output file
model_path ="/root/dpo/.cache/root/dataset_DPO_loss_dpo_pythia28_2024-05-22_21-25-05_438706/LATEST/model.pt"
output_path = "/root/dpo/.cache/root/dataset_DPO_loss_dpo_pythia28_2024-05-22_21-25-05_438706/LATEST/pytorch_model.bin"

# Load the state dictionary from the .pt file
state_dict = torch.load(model_path)

# Save the state dictionary to pytorch_model.bin
torch.save(state_dict, output_path)

print(f"Model successfully converted and saved to {output_path}")

