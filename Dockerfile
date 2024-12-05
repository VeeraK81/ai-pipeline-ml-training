# Use Miniconda as the base image to create a clean Python environment
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /home

# Copy the current directory contents into the container at /home
COPY . .

# Set the PYTHONPATH environment variable for proper imports
ENV PYTHONPATH=/home

# Update apt-get and install necessary packages: nano, unzip, curl, and Python packages
RUN apt-get update && apt-get install -y \
    nano \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*  # Clean up apt cache to reduce image size

# Install the Deta CLI using the curl script
RUN curl -fsSL https://get.deta.dev/cli.sh | sh


# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Command to run the application (adjust if your entry point is different)
CMD ["python", "app/ai_solution_ml_train.py"]


