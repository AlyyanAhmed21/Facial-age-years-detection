# research/evaluation.py

import torch
import pandas as pd
import json
from pathlib import Path
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the project's src directory to the Python path
# This allows us to import our custom modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

# Now we can import our custom classes
from cnnClassifier.components.multi_task_model_trainer import MultiTaskEfficientNet, FairFaceDataset
from cnnClassifier.utils.common import read_yaml
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoImageProcessor

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Define paths directly. We are not using the config manager.
MODEL_PATH = Path("artifacts/multi_task_model_trainer/facial_demographics_model")
DATA_PATH = Path("artifacts/data_preparation/fairface_cleaned.csv")
PARAMS_PATH = Path("params.yaml")
EVALUATION_OUTPUT_DIR = Path("artifacts/manual_evaluation")
EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load parameters
params = read_yaml(PARAMS_PATH)
IMAGE_SIZE = params.IMAGE_SIZE
BATCH_SIZE = params.BATCH_SIZE
TEST_SPLIT_SIZE = params.TEST_SPLIT_SIZE
RANDOM_STATE = params.RANDOM_STATE

# ==============================================================================
# MAIN EVALUATION LOGIC
# ==============================================================================
def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running evaluation on device: {device} ---")

    # 1. Load data and prepare the test split
    print("Loading and preparing test data...")
    df = pd.read_csv(DATA_PATH)
    
    label_maps = {}
    for task in ['age', 'gender', 'race']:
        labels = sorted(df[task].unique())
        label_maps[f'{task}_label2id'] = {label: i for i, label in enumerate(labels)}
        label_maps[f'{task}_id2label'] = {i: label for i, label in enumerate(labels)}
        df[f'{task}_id'] = df[task].map(label_maps[f'{task}_label2id'])
    
    # Use the same random_state to ensure we get the identical test split as in training
    _, test_df = train_test_split(
        df, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=df['age']
    )
    
    # 2. Create the PyTorch DataLoader
    model_config = read_yaml(Path("config/config.yaml"))
    base_model_name = model_config.multi_task_model_trainer.model_name
    processor = AutoImageProcessor.from_pretrained(base_model_name)
    _transforms = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    test_dataset = FairFaceDataset(dataframe=test_df, processor=processor, transforms=_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Load the trained model
    print(f"Loading model from: {MODEL_PATH}")
    model = MultiTaskEfficientNet(
        model_name=str(MODEL_PATH), # Pass the path as the model name
        num_labels_age=len(label_maps['age_id2label']),
        num_labels_gender=len(label_maps['gender_id2label']),
        num_labels_race=len(label_maps['race_id2label']),
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH / 'pytorch_model.bin', map_location=device))
    model.eval()

    # 4. Run predictions on the test set
    print("Running predictions on the test set...")
    all_preds = {'age': [], 'gender': [], 'race': []}
    all_labels = {'age': [], 'gender': [], 'race': []}

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels']
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        
        all_preds['age'].extend(outputs['age_logits'].argmax(1).cpu().numpy())
        all_preds['gender'].extend(outputs['gender_logits'].argmax(1).cpu().numpy())
        all_preds['race'].extend(outputs['race_logits'].argmax(1).cpu().numpy())

        all_labels['age'].extend(labels['age'].cpu().numpy())
        all_labels['gender'].extend(labels['gender'].cpu().numpy())
        all_labels['race'].extend(labels['race'].cpu().numpy())

    # 5. Calculate metrics, generate reports, and save artifacts
    print("--- Evaluation Results ---")
    metrics = {}
    for task in ['age', 'gender', 'race']:
        accuracy = accuracy_score(all_labels[task], all_preds[task])
        print(f"\n--- {task.capitalize()} ---")
        print(f"Accuracy: {accuracy:.4f}")
        
        report_str = classification_report(
            all_labels[task], 
            all_preds[task], 
            target_names=list(label_maps[f'{task}_id2label'].values())
        )
        print("Classification Report:")
        print(report_str)
        
        metrics[f'{task}_accuracy'] = accuracy
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels[task], all_preds[task])
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(label_maps[f'{task}_id2label'].values()), yticklabels=list(label_maps[f'{task}_id2label'].values()), cmap='Blues')
        plt.title(f'Confusion Matrix for {task.capitalize()}', fontsize=16)
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        cm_path = EVALUATION_OUTPUT_DIR / f'{task}_confusion_matrix.png'
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close() # Close the plot to avoid displaying it in the console
        print(f"Saved {task} confusion matrix to {cm_path}")

    # Save metrics to a JSON file
    metrics_path = EVALUATION_OUTPUT_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved final metrics to {metrics_path}")
    
    # Save the label maps used for this evaluation run
    label_maps_path = EVALUATION_OUTPUT_DIR / "label_maps.json"
    with open(label_maps_path, 'w') as f:
        json.dump(label_maps, f, indent=4)
    print(f"Saved label maps to {label_maps_path}")


if __name__ == '__main__':
    evaluate_model()