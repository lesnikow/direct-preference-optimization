FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    neovim \
    htop \
    atop \
    tree \
    python3.10 \
    python3.10-venv \
    git \
    zsh

# Create a virtual environment
RUN python3 -m venv /opt/env

# Set environment variables to activate the virtual environment
ENV VIRTUAL_ENV=/opt/env
ENV PATH="/opt/env/bin:$PATH"

# Clone the repository
RUN git clone https://github.com/lesnikow/direct-preference-optimization.git /app

# Set the working directory
WORKDIR /app

# Install requirements and upgrade datasets
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --upgrade datasets

# Install and log in to wandb
RUN pip install wandb && \
    wandb login 3df7ad506a96b198d251a4df07f7c9b5bd4745e3

# Set zsh as the default shell
SHELL ["/usr/bin/zsh", "-c"]

# Set the default command to run when the container starts
CMD ["zsh"]
