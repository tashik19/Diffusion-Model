# Use the official PyTorch image as the base
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY . .

# Command to run on container start
CMD ["python", "./fall.py"]
