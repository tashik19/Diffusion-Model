import numpy as np
import torch
import pandas as pd
import xlsxwriter
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

import os
import openpyxl

# Directory containing the Excel files
excel_dir = 'Data/'

# Get a list of all Excel files in the directory
excel_files = [file for file in os.listdir(excel_dir) if file.endswith('.xlsx')]

# Split the data into trials based on empty spaces
trials = []
current_trial = []

for excel_file in excel_files:
    # Construct the full path to the Excel file
    excel_file_path = os.path.join(excel_dir, excel_file)

    try:
        # Open the Excel file and read sheet names
        excel_workbook = openpyxl.load_workbook(excel_file_path, read_only=True)
        sheet_names = excel_workbook.sheetnames

        # Find the sheet with a name that ends with "trimmed"
        for sheet_name in sheet_names:
            if sheet_name.endswith("trimmed"):
                break  # Found the desired sheet
        else:
            # Handle the case when no matching sheet is found
            print(f"No sheet ending with 'trimmed' found in {excel_file}")
            continue

        # Load the data from the selected sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

        # Extract accelerometer data columns (x, y, z)
        accelerometer_data = df[['x-axis (g)', 'y-axis (g)', 'z-axis (g)']].values

        for row in accelerometer_data:
            if np.all(np.isnan(row)):
                if current_trial:
                    trials.append(current_trial)
                    current_trial = []
            else:
                current_trial.append(row)

        if current_trial:
            trials.append(current_trial)

        # Process 'trials' as needed for each file
        # You can perform additional operations or save the trials data for each file here

    except Exception as e:
        print(f"An error occurred while processing {excel_file}: {e}")

# Calculate the desired trial length as a multiple of 1, 2, 4, or 8
highest_trial_length = max(len(trial) for trial in trials)
desired_trial_length = highest_trial_length

print(desired_trial_length)

while desired_trial_length % 8 != 0:
    desired_trial_length += 1

print(desired_trial_length)

# Pad each trial with zeros to the desired length
padded_trials = []

for trial in trials:
    while len(trial) < desired_trial_length:
        trial.append(trial[-1])
    padded_trials.append(trial)

print(padded_trials)

# Convert the data into a 3D tensor
tensor_data = np.array(padded_trials)

# Reshape the tensor to match your desired dimensions
tensor_data = tensor_data.transpose(0, 2, 1)

training_seq = torch.tensor(tensor_data)

print(training_seq.shape)

print(training_seq)

training_seq = training_seq.float()

print(training_seq.dtype)

model = Unet1D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3  # 3 columns
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=desired_trial_length,  # number of data in 1 trial. this should be divided by 1, 2, 4, 8
    timesteps=1000,
    objective='pred_v'
)

dataset = Dataset1D(
    training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

loss = diffusion(training_seq)
loss.backward()

# # Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 70,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()


sampled_seq = diffusion.sample(batch_size=20)
sampled_seq.shape  # (4, 32, 128) // 4 trials, 32 features, 128 rows in one trial

print(sampled_seq)

tensor_data = np.array(sampled_seq)

excel_file_path = "trial_data.xlsx"
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    for trial_number, trial_data in enumerate(tensor_data, start=1):
        # Reshape the trial data
        reshaped_data = trial_data.T  # Transpose to have features as columns

        # Create a Pandas DataFrame for the trial data
        df = pd.DataFrame(reshaped_data, columns=[f"Feature{i}" for i in range(1, reshaped_data.shape[1] + 1)])

        # Add the DataFrame to the Excel file with the trial number as the sheet name
        df.to_excel(writer, sheet_name=f"Trial_{trial_number}", index=False)