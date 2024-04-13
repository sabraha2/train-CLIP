# Job Submission Script for Model Training

This directory contains the `train_job_submission.sh` script, which is a job submission script for training the CLIP model on a high-performance computing (HPC) cluster using a job scheduler like Sun Grid Engine (SGE).

## Script Description

The `train_job_submission.sh` script automates the process of setting up the environment, activating the Python virtual environment, and running the training script with the specified parameters.

## Usage

Before using the script, make sure to:
- Replace the placeholder email `your_email@domain.com` with your actual email to receive job notifications.
- Modify the `source` command with the path to your Python virtual environment.
- Adjust the script parameters like `--folder`, `--batch_size`, `--num_workers`, and `--default_root_dir` according to your dataset and desired configuration.

To submit a training job to the queue, simply execute the following command:

```bash
qsub train_job_submission.sh


# Model Evaluation

This directory contains the evaluation script `evaluate_clip_model.py` for the CLIP model.

## Overview

The script `evaluate_clip_model.py` is designed to evaluate the performance of the trained CLIP model on a specified test dataset. It computes image and text embeddings and visualizes them using t-SNE plots. It also visualizes the distribution of cosine similarities between image and text embeddings.

## Prerequisites

Ensure all dependencies are installed as specified in the main `requirements.txt`, including `torch`, `transformers`, `matplotlib`, `numpy`, and `scikit-learn`.

## Usage

Run the script with the following command:

```bash
python evaluate_clip_model.py --model_checkpoint_path PATH_TO_CHECKPOINT --test_dataset_path PATH_TO_TEST_DATASET --batch_size BATCH_SIZE
