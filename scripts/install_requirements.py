# import libraries
import sys
import os

# append the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '../')
req_path = os.path.join(root_dir, 'requirements.txt')

import subprocess

# Specify the name of the Conda environment and Python version
env_name = "env-diabetes"
python_version = "3.10"

# Check if the environment already exists
existing_envs = subprocess.check_output(["conda", "env", "list"]).decode('utf-8')
if env_name not in existing_envs:
    # Create the Conda environment if it doesn't exist
    subprocess.run(["conda", "create", "--name", env_name, f"python={python_version}", "--yes"], check=True)
else:
    print(f"Environment '{env_name}' already exists.")

# Read packages from requirements.txt
with open(req_path, "r") as f:
    packages = f.read().splitlines()

# Install packages using Conda
for package in packages:
    subprocess.run(["conda", "install", "--name", env_name, package, "--yes"], check=True)

print(f"Environment '{env_name}' created and packages installed.")
    
