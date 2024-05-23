import torch

# Define the path to the input model and the output file
exp_dir = "dataset_DCPO_loss_dpo_pythia28_2024-05-22_21-45-42_170746"
model_path = f"/root/dpo/.cache/root/{exp_dir}/LATEST/policy.pt"
output_path = f"/root/dpo/.cache/root/{exp_dir}/LATEST/pytorch_model.bin"

# Load the state dictionary from the .pt file
state_dict = torch.load(model_path)

# Save the state dictionary to pytorch_model.bin
torch.save(state_dict, output_path)

print(f"Model successfully converted and saved to {output_path}")

