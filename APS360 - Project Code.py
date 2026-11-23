import sys  # Inspect the Python interpreter used by this notebook.
import platform  # Identify the operating system of the runtime.
from pathlib import Path  # Work with local file and folder paths.

print('Operating system:', platform.system())  # Windows users should see Windows when connected locally.
print('Python executable:', sys.executable)  # Show the Python environment running the code.
print('Working directory:', Path.cwd())  # Report the current folder on the runtime's disk.
if Path('/content').is_dir() and platform.system() == 'Linux':  # A common sign of a hosted Colab runtime, not definitive.
    print('If this is a hosted Colab runtime, reconnect to your local runtime before using local paths.')  # Explain how to access the computer's disk.

from __future__ import annotations  # Postpone evaluation of type hints to improve compatibility.

import argparse  # Read command-line arguments such as dataset location and epochs.
import copy  # Make an independent snapshot of the best model weights.
import json  # Save model metrics and experiment settings as JSON.
import random  # Seed Python's random generator for reproducibility.
import warnings  # Issue warnings about incomplete or unexpected input data.
from pathlib import Path  # Use cross-platform objects for paths to files and folders.

import numpy as np  # Use NumPy arrays for targets, thresholds, and statistics.
import pandas as pd  # Use pandas to read the NIH metadata CSV and save result tables.
import torch  # Use PyTorch tensors, devices, backpropagation, and optimizers.
from PIL import Image  # Load and convert medical images using Pillow.
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # Import evaluation measures for imbalanced multi-label classification.
from torch import nn  # Access neural network layers and loss functions.
from torch.utils.data import DataLoader, Dataset  # Define datasets and load them in minibatches.
from torchvision import models, transforms  # Import pretrained vision architectures and image transformations.
from torchvision.transforms import InterpolationMode  # Choose bilinear interpolation while resizing and rotating images.


# The NIH dataset has 14 SEPARATE labels called "Nodule" and "Mass" (not one
# label called "Nodule Mass"). Its CSV spells Pleural_Thickening with '_'.
CONDITIONS = [  # Keep the 14 prediction outputs in the exact order of this list.
    "Atelectasis",  # Disease label 1 of 14; this position defines its output index.
    "Cardiomegaly",  # Disease label 2 of 14; this position defines its output index.
    "Consolidation",  # Disease label 3 of 14; this position defines its output index.
    "Edema",  # Disease label 4 of 14; this position defines its output index.
    "Effusion",  # Disease label 5 of 14; this position defines its output index.
    "Emphysema",  # Disease label 6 of 14; this position defines its output index.
    "Fibrosis",  # Disease label 7 of 14; this position defines its output index.
    "Hernia",  # Disease label 8 of 14; this position defines its output index.
    "Infiltration",  # Disease label 9 of 14; this position defines its output index.
    "Mass",  # Disease label 10 of 14; this position defines its output index.
    "Nodule",  # Disease label 11 of 14; this position defines its output index.
    "Pleural_Thickening",  # Disease label 12 of 14; this position defines its output index.
    "Pneumonia",  # Disease label 13 of 14; this position defines its output index.
    "Pneumothorax",  # Disease label 14 of 14; this position defines its output index.
]  # End of the ordered list of target conditions.
NUM_CLASSES = len(CONDITIONS)  # Count labels so all classifier heads output 14 separate scores.

"""## 3. Experiment configuration (edit this cell)

Set `DATA_ROOT` to the directory on your own hard drive that contains the NIH metadata and images. Windows paths use raw strings (`r"..."`) to avoid backslash escape errors. You can also specify a separate image root or results folder. Keep `num_workers=0` for reliable Windows notebook execution. All training settings and model options are preserved from the previous notebook.
"""

# Configure dataset folders; edit the paths if you use a different structure.
DATA_ROOT = Path(r"C:\Users\colin\Downloads\archive")  # CHANGE this example to the actual NIH dataset folder on your hard drive.
CSV_PATH = DATA_ROOT / 'Data_Entry_2017.csv'  # NIH CSV with image names, patient IDs, and disease labels.
IMAGES_DIR = DATA_ROOT  # Search this directory recursively for all extracted X-ray images.
OUTPUT_DIR = DATA_ROOT.parent / 'APS360_results'  # Save checkpoints, logs, and CSVs beside the dataset on your own drive.

args = argparse.Namespace(  # Use an object with the same settings as the original script's command-line parser.
    csv=CSV_PATH,  # Location of Data_Entry_2017.csv.
    images_dir=IMAGES_DIR,  # Root directory holding the extracted image files.
    models=['small_cnn'],  # Initializtion of small CNN
    output_dir=OUTPUT_DIR,  # Save the results for each architecture under this folder.
    epochs=10,  # Train up to 10 passes through the training data.
    batch_size=16,  # Process 16 images per training minibatch.
    num_workers=0,  # Use zero workers for compatibility with Windows/Jupyter notebooks.
    lr_densenet=1e-4,  # Adam learning rate for DenseNet-121.
    lr_baseline=1e-3,  # Adam learning rate for the CNN and logistic regression.
    seed=42,  # Use the same random seed for splitting and initialization.
    patience=4,  # Stop after 4 epochs without validation improvement (0 disables stopping).
    pos_weight=False,  # When True, apply the same positive-class weighting rule to every model.
    freeze_backbone=False,  # When True, train only DenseNet's final classification head.
    no_pretrained=False,  # False means initialize DenseNet with ImageNet-pretrained weights.
)  # Finish the configuration object.

print('CSV:', args.csv)
print('Image root:', args.images_dir)
print('Results:', args.output_dir)
print('Models:', args.models)

"""## 4. Dataset loader and network architectures

The dataset yields grayscale images and a 14-element multi-hot label vector. DenseNet converts grayscale to three repeated channels to use pretrained ImageNet weights; the two baselines use one grayscale channel.
"""

class NIHChestXrayDataset(Dataset):  # Adapt the PyTorch Dataset interface to NIH chest X-rays.
    """Load NIH grayscale X-rays, returning (image_tensor, 14-label tensor)."""

    def __init__(self, metadata: pd.DataFrame, transform: transforms.Compose):  # Initialize the dataset with metadata rows and a transform pipeline.
        self.paths = metadata["image_path"].tolist()  # Save file paths so each sample can load the correct image.
        self.targets = np.stack(metadata["targets"].to_numpy()).astype(np.float32)  # Stack 14-label vectors into a float32 array of shape (N, 14).
        self.transform = transform  # Remember how each image should be preprocessed.

    def __len__(self) -> int:  # Tell DataLoader how many X-ray samples are available.
        return len(self.paths)  # Return the number of stored image paths.

    def __getitem__(self, index: int):  # Define how to fetch a single indexed training or test sample.
        with Image.open(self.paths[index]) as im:  # Open the selected image and close it safely afterward.
            image = self.transform(im.convert("L"))  # Convert the image to grayscale and apply resize/augmentation/normalization.
        return image, torch.from_numpy(self.targets[index])  # Return an image tensor and its 14 binary disease targets.

class DenseNetChestXray(nn.Module):  # Define the proposal's principal DenseNet-121 classifier.
    """Pretrained DenseNet-121 with a 14-logit multi-label classification head."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):  # Allow ImageNet pretraining and optional freezing of feature layers.
        super().__init__()  # Initialize the base PyTorch neural-network module.
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None  # Select ImageNet weights unless pretraining was disabled.
        self.network = models.densenet121(weights=weights)  # Construct the DenseNet-121 architecture with the chosen weights.
        in_features = self.network.classifier.in_features  # Determine the number of features entering DenseNet's original head.
        self.network.classifier = nn.Linear(in_features, NUM_CLASSES)  # Replace its classifier with a fully connected 14-logit layer.
        self.freeze_backbone = freeze_backbone  # Record whether the pretrained feature extractor should be frozen.
        if freeze_backbone:  # Check whether only the classification head should be trained.
            self.network.features.requires_grad_(False)  # Disable gradient updates in all DenseNet feature layers.

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Describe how input images propagate through DenseNet.
        return self.network(x)  # Return 14 raw logits; sigmoid belongs in probability inference, not before BCEWithLogitsLoss.


class SmallCNNBaseline(nn.Module):  # Define a simpler CNN baseline trained without ImageNet weights.
    """Unpretrained 3-block CNN: spatial features, global pooling, 14 logits."""

    def __init__(self):  # Initialize the three-block CNN architecture.
        super().__init__()  # Initialize the base PyTorch module before defining layers.
        self.features = nn.Sequential(  # Build a sequential feature extractor from convolution and pooling blocks.
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Detect 16 local patterns with 3x3 filters from one grayscale channel.
            nn.ReLU(inplace=True),  # Apply a nonlinear activation so the network learns complex features.
            nn.MaxPool2d(2),  # Halve image width and height: 224 x 224 -> 112 x 112.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Expand from 16 to 32 feature maps with a second convolution.
            nn.ReLU(inplace=True),  # Add a second nonlinear activation.
            nn.MaxPool2d(2),  # Halve spatial dimensions again: 112 x 112 -> 56 x 56.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Expand from 32 to 64 feature maps in the third convolution.
            nn.ReLU(inplace=True),  # Add the third nonlinear activation.
            nn.MaxPool2d(2),  # Downsample feature maps from 56 x 56 -> 28 x 28.
            nn.AdaptiveAvgPool2d((1, 1)),  # Average each of 64 feature maps to a single number (global pooling).
        )  # Finish the sequential feature extractor.
        self.classifier = nn.Linear(64, NUM_CLASSES)  # Convert 64 extracted features into 14 independent disease logits.

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Specify the baseline CNN's forward computation.
        features = self.features(x).flatten(1)  # Extract features and flatten shape (batch,64,1,1) into (batch,64).
        return self.classifier(features)  # Produce one independent raw disease logit per class.


class LogisticBaseline(nn.Module):  # Define linear logistic regression without any convolutional layers.
    """Logistic regression directly on 64x64 (downsampled) grayscale pixels."""

    def __init__(self):  # Initialize the logistic baseline.
        super().__init__()  # Initialize the parent neural-network module.
        self.classifier = nn.Linear(64 * 64, NUM_CLASSES)  # Map 4,096 grayscale pixel intensities directly to 14 logits.

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Specify how a batch of 64x64 images is classified.
        return self.classifier(x.flatten(start_dim=1))  # Flatten pixels, then predict 14 disease logits with a linear layer.

def build_model(name: str, args: argparse.Namespace) -> nn.Module:  # Factory: select and construct the requested architecture by its name.
    if name == "densenet":  # Check whether the main DenseNet classifier was requested.
        return DenseNetChestXray(  # Create the main pretrained or unpretrained DenseNet model.
            pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone  # Forward CLI flags determining whether to preload or freeze weights.
        )  # Complete construction of the DenseNet classifier.
    if name == "small_cnn":  # Check whether the smaller CNN baseline was requested.
        return SmallCNNBaseline()  # Return a new CNN whose parameters start untrained.
    if name == "logistic":  # Check whether the linear pixel-level baseline was requested.
        return LogisticBaseline()  # Return a new multi-label logistic model.
    raise ValueError(f"Unknown model: {name}")  # Reject misspelled or unsupported model identifiers.


def make_transform(model_name: str, train: bool) -> transforms.Compose:  # Create image transforms specific to each model and split.
    """Retain grayscale information. DenseNet's RGB input repeats the gray image."""
    size = 64 if model_name == "logistic" else 224  # Use smaller images for the pixel-level baseline; 224 for CNN/DenseNet.
    pipeline = [transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR)]  # Resize each X-ray with bilinear interpolation.
    if train:  # Enable random rotations only for training, not evaluation.
        # Proposal specifies augmentation with +/- 10-degree image rotation.
        pipeline.append(  # Add a transform to the training preprocessing pipeline.
            transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR, fill=0)  # Rotate randomly within +/-10 degrees; use black fill at exposed edges.
        )  # Finish adding the random-rotation transform.
    if model_name == "densenet":  # Check whether DenseNet needs three-channel input.
        # ImageNet-pretrained weights need 3 channels and ImageNet normalization.
        # Grayscale is replicated, so no new colour information is introduced.
        pipeline.append(transforms.Grayscale(num_output_channels=3))  # Replicate grayscale values in three channels for pretrained DenseNet.
    pipeline.append(transforms.ToTensor())  # Convert the transformed PIL image to a float tensor in [0, 1].
    if model_name == "densenet":  # Check whether to apply DenseNet's ImageNet normalization.
        pipeline.append(  # Append the channel-wise normalization operation.
            transforms.Normalize(  # Create a Normalize transform with RGB channel parameters.
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # Apply the mean and standard deviation expected by ImageNet weights.
            )  # Complete ImageNet normalization settings.
        )  # Finish appending the normalization operation.
    return transforms.Compose(pipeline)  # Combine all preprocessing operations in the specified order.

"""## 6. Match NIH images with labels and split by patient

Files are matched to the metadata CSV by `Image Index`. A single patient is assigned to only one split, preventing a patient's separate X-rays from appearing in both training and evaluation.
"""

def find_image_paths(images_dir: Path) -> dict[str, str]:  # Map each image filename to its actual location on disk.
    """Index PNGs in all NIH images_*/* directories (or in one flat folder)."""
    if not images_dir.is_dir():  # Check that the user-provided image directory exists.
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")  # Stop with an informative error if the folder is missing.
    result = {}  # Prepare a dictionary for filename-to-filepath lookup.
    for path in images_dir.rglob("*"):  # Search all nested folders under the extracted NIH directory.
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:  # Skip folders and files that are not supported images.
            continue  # Continue to the next path without indexing it.
        if path.name in result:  # Detect duplicate image filenames that would make matching ambiguous.
            raise ValueError(f"Duplicate image filename found: {path.name}")  # Stop instead of silently choosing an incorrect duplicate.
        result[path.name] = str(path)  # Store the resolved path using the image filename as key.
    if not result:  # Check whether the recursive search found any supported image files.
        raise FileNotFoundError(  # Report the missing-data error if the image tree is empty.
            f"No images found under {images_dir}. Extract the NIH images_*.tar.gz archives."  # Explain that the NIH image archives must be extracted first.
        )  # Complete the missing-image error message.
    print(f"Indexed {len(result):,} images under {images_dir}")  # Report how many image files were indexed.
    return result  # Return the dictionary for matching CSV rows to image files.


def load_metadata(csv_path: Path, images_dir: Path) -> pd.DataFrame:  # Read NIH labels, verify inputs, and build per-image target vectors.
    df = pd.read_csv(csv_path)  # Load the NIH CSV into a pandas DataFrame.
    required = {"Image Index", "Finding Labels", "Patient ID"}  # Declare the metadata columns required for the project.
    missing_cols = required.difference(df.columns)  # Identify any required columns absent from the CSV.
    if missing_cols:  # Test whether essential metadata columns are missing.
        raise ValueError(f"Metadata CSV is missing required columns: {sorted(missing_cols)}")  # Raise a readable error naming each missing column.
    if df[["Image Index", "Patient ID"]].isna().any().any():  # Check that image names and patient IDs have no null entries.
        raise ValueError("Image Index and Patient ID must not be missing")  # Reject rows whose identity information is unusable.

    image_by_name = find_image_paths(images_dir)  # Collect the actual image paths from the extracted dataset.
    df["image_path"] = df["Image Index"].map(image_by_name)  # Match the CSV's Image Index names to indexed file locations.
    missing_images = int(df["image_path"].isna().sum())  # Count CSV rows with no matching image files.
    if missing_images:  # Check whether some metadata refers to absent X-rays.
        warnings.warn(  # Warn that some examples will be discarded.
            f"{missing_images:,} of {len(df):,} CSV rows have no extracted image "  # Start a warning showing how many X-ray files were missing.
            "and will be excluded. Complete the dataset download for full results."  # Explain why fewer rows will be used than are listed in the CSV.
        )  # Complete the warning about unmatched image files.
    df = df.dropna(subset=["image_path"]).copy()  # Keep only metadata rows for which a real image was found.
    if len(df) < 3:  # Check whether enough images remain for a minimal experiment.
        raise ValueError("Too few matching images for training")  # Stop if fewer than three images can be matched.

    known = set(CONDITIONS)  # Construct a lookup set of the 14 supported condition names.
    unknown = set()  # Track unsupported labels for reporting.
    targets = []  # Collect one 14-element numeric target vector per image.
    for label_string in df["Finding Labels"].fillna("No Finding").astype(str):  # Read each pipe-separated diagnosis string; fill blank diagnoses as No Finding.
        labels = {part.strip().replace("Pleural Thickening", "Pleural_Thickening")  # Normalize alternate spelling before making a set of diagnosis names.
                  for part in label_string.split("|")}  # Split the diagnosis field on the NIH pipe separator.
        unknown.update(labels - known - {"No Finding"})  # Record any unrecognized diagnoses, excluding No Finding.
        targets.append(np.asarray([float(c in labels) for c in CONDITIONS], dtype=np.float32))  # Encode present diseases as 1 and absent diseases as 0.
    if unknown:  # Check whether labels outside the expected 14 were seen.
        warnings.warn(f"Labels not present in the 14 NIH classes: {sorted(unknown)}")  # Warn that any unknown labels are not modeled.
    df["targets"] = targets  # Attach the target array to the matching DataFrame row.
    return df.reset_index(drop=True)  # Return indexed, clean metadata for all available images.


def split_by_patient(df: pd.DataFrame, seed: int):  # Create train/validation/test groups with no shared patient IDs.
    """75/15/10 split by patient IDs; avoids train/test patient leakage."""
    rng = np.random.default_rng(seed)  # Create a reproducible random number generator.
    patients = rng.permutation(df["Patient ID"].unique())  # Shuffle unique patient IDs rather than individual image rows.
    if len(patients) < 20:  # Check there are enough patients for three meaningful partitions.
        raise ValueError("At least 20 distinct patients are needed for this split")  # Reject tiny inputs with fewer than 20 distinct patients.
    n_test = max(1, round(0.10 * len(patients)))  # Allocate about 10% of patients to the held-out test set.
    n_val = max(1, round(0.15 * len(patients)))  # Allocate about 15% of patients to validation.
    test_patients = set(patients[:n_test])  # Select shuffled patient IDs for final testing.
    val_patients = set(patients[n_test:n_test + n_val])  # Select a disjoint set of IDs for validation.
    train_patients = set(patients[n_test + n_val:])  # Put all remaining patient IDs in the training set.

    train = df[df["Patient ID"].isin(train_patients)].copy()  # Keep images belonging only to training patients.
    val = df[df["Patient ID"].isin(val_patients)].copy()  # Keep images belonging only to validation patients.
    test = df[df["Patient ID"].isin(test_patients)].copy()  # Keep images belonging only to testing patients.
    assert len(train) + len(val) + len(test) == len(df)  # Verify that the split includes every original image exactly once.
    assert not (train_patients & val_patients or train_patients & test_patients  # Assert there is no patient overlap between training and evaluation groups.
                or val_patients & test_patients)  # Check the remaining overlap between validation and test groups.
    for split_name, subset in (("train", train), ("val", val), ("test", test)):  # Iterate through the split summaries in a stable order.
        print(f"{split_name:>5}: {len(subset):>7,} images; "  # Start printing the image count for this split.
              f"{subset['Patient ID'].nunique():>6,} patients")  # Also print the number of unique patients in this split.
    return train, val, test  # Return the three separate metadata tables.


def get_loader(df: pd.DataFrame, name: str, training: bool, args, device):  # Make a batch loader with the transforms needed by this model.
    dataset = NIHChestXrayDataset(df, transform=make_transform(name, train=training))  # Wrap the split DataFrame in the custom X-ray Dataset.
    return DataLoader(  # Construct a PyTorch DataLoader for minibatch iteration.
        dataset,  # Supply the Dataset that will provide each image and label vector.
        batch_size=args.batch_size,  # Use the requested batch size, such as 16 X-rays at a time.
        shuffle=training,  # Shuffle batches during training but not during validation or testing.
        num_workers=args.num_workers,  # Configure background worker processes for file loading.
        pin_memory=(device.type == "cuda"),  # Pin host memory when transferring minibatches to a CUDA GPU.
    )  # Finish constructing the DataLoader.

def train_one_epoch(model, loader, loss_fn, optimizer, device, freeze_backbone=False):  # Train the model over one complete pass through its minibatches.
    model.train()  # Enable training behaviors such as dropout and BatchNorm updates.
    if freeze_backbone:  # Check whether DenseNet's feature extractor should remain frozen.
        # Keep running BatchNorm statistics frozen too, not just the parameters.
        model.network.features.eval()  # Prevent pretrained BatchNorm statistics from changing when frozen.
    total_loss = 0.0  # Initialize the running sum of image-weighted training loss.
    total_items = 0  # Initialize the number of images processed in this epoch.
    for images, labels in loader:  # Iterate over image and target batches from the training loader.
        images = images.to(device, non_blocking=True)  # Move the input images onto the selected CPU or accelerator.
        labels = labels.to(device, non_blocking=True)  # Move the 14-label targets to the same device as model outputs.
        optimizer.zero_grad(set_to_none=True)  # Erase gradients from the previous minibatch.
        logits = model(images)  # Compute 14 raw prediction scores for every input image.
        loss = loss_fn(logits, labels)  # Compare predicted logits to the 0/1 targets using BCEWithLogitsLoss.
        loss.backward()  # Backpropagate the error to compute parameter gradients.
        optimizer.step()  # Perform an Adam parameter update using those gradients.
        batch_items = images.size(0)  # Determine the actual batch size (the last batch may be smaller).
        total_loss += float(loss.item()) * batch_items  # Accumulate batch loss weighted by its number of images.
        total_items += batch_items  # Count all images processed this epoch.
    return total_loss / max(1, total_items)  # Return average per-image training loss; avoid division by zero.


@torch.no_grad()  # Disable gradient tracking to save memory during evaluation.
def predict(model, loader, loss_fn, device):  # Calculate average loss, true labels, and probabilities for a split.
    model.eval()  # Switch to evaluation mode to disable training-only behavior.
    labels_out, probs_out = [], []  # Prepare lists to collect outputs from successive minibatches.
    total_loss = 0.0  # Start an accumulator for evaluation loss.
    total_items = 0  # Start an accumulator for processed evaluation samples.
    for images, labels in loader:  # Visit every image batch without shuffling predictions.
        images = images.to(device, non_blocking=True)  # Move evaluation images to the chosen computation device.
        labels = labels.to(device, non_blocking=True)  # Move evaluation labels onto the same device.
        logits = model(images)  # Compute per-disease logits without updating parameters.
        batch_items = images.size(0)  # Count the number of images in this minibatch.
        total_loss += float(loss_fn(logits, labels).item()) * batch_items  # Accumulate batch BCE loss, weighted by image count.
        total_items += batch_items  # Accumulate the total number of images.
        labels_out.append(labels.cpu().numpy())  # Copy true labels to CPU NumPy arrays for metric calculations.
        probs_out.append(torch.sigmoid(logits).cpu().numpy())  # Apply sigmoid to logits, yielding independent probabilities in [0,1].
    return (total_loss / max(1, total_items),  # Return the average loss together with concatenated outputs.
            np.concatenate(labels_out), np.concatenate(probs_out))  # Stack every minibatch into (N,14) arrays for labels and probabilities.

def choose_thresholds(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:  # Select a separate binary classification threshold per condition.
    """Choose independent F1-maximizing cutoffs using VALIDATION data only."""
    thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float64)  # Start with a decision threshold of 0.5 for every condition.
    choices = np.linspace(0.05, 0.95, 19)  # Define candidate thresholds from 0.05 to 0.95 in steps of 0.05.
    for j in range(NUM_CLASSES):  # Optimize the threshold for one disease label at a time.
        if y_true[:, j].sum() == 0:  # Check whether validation contains any positive images for this disease.
            continue  # Keep the default threshold if there are no positive examples.
        scores = [f1_score(y_true[:, j], probs[:, j] >= t, zero_division=0)  # Calculate the F1 score for each candidate cutoff.
                  for t in choices]  # Repeat that score calculation for all threshold options.
        thresholds[j] = float(choices[int(np.argmax(scores))])  # Retain the candidate threshold giving maximum validation F1.
    return thresholds  # Return the selected 14-element threshold array.


def compute_metrics(y_true: np.ndarray, probs: np.ndarray,  # Accept true labels and probabilities to compute classification metrics.
                    thresholds: np.ndarray) -> dict:  # Receive one decision threshold for each of 14 conditions.
    per_class = {}  # Prepare a dictionary holding metrics by condition.
    valid_aucs = []  # Collect disease ROC-AUC values that are mathematically defined.
    valid_aps = []  # Collect disease average-precision values that are defined.
    for j, condition in enumerate(CONDITIONS):  # Process all 14 conditions in the NIH label order.
        labels = y_true[:, j].astype(int)  # Extract the true zero/one vector for the current disease.
        predicted = (probs[:, j] >= thresholds[j]).astype(int)  # Convert probabilities into predictions using that disease's cutoff.
        if np.unique(labels).size == 2:  # Require both a positive and a negative example to compute ROC-AUC.
            auc = float(roc_auc_score(labels, probs[:, j]))  # Calculate area under the receiver-operating characteristic curve.
            ap = float(average_precision_score(labels, probs[:, j]))  # Calculate average precision from the precision-recall curve.
            valid_aucs.append(auc)  # Include this class in macro ROC-AUC.
            valid_aps.append(ap)  # Include this class in macro average precision.
        else:  # Handle classes that have only one label value in this split.
            # ROC-AUC is undefined without both positive and negative samples.
            auc = None  # Mark undefined ROC-AUC as null in JSON instead of an invalid number.
            ap = None  # Mark unavailable average precision as null for consistency.
        per_class[condition] = {  # Open the per-class results dictionary for this condition.
            "positive_images": int(labels.sum()),  # Count ground-truth positive images, not predicted true positives.
            "roc_auc": auc,  # Store this condition's ROC-AUC or null.
            "average_precision": ap,  # Store this condition's average precision or null.
            "f1": float(f1_score(labels, predicted, zero_division=0)),  # Calculate the condition-specific F1 score at the chosen cutoff.
            "decision_threshold": float(thresholds[j]),  # Save the probability threshold used to make binary predictions.
        }  # Close the metrics dictionary for this condition.
    predictions = (probs >= thresholds.reshape(1, -1)).astype(int)  # Generate a binary (N,14) prediction matrix for micro F1.
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))  # Average the F1 scores over all 14 conditions equally.
    return {  # Assemble dataset-level metrics and the full per-class results.
        "macro_roc_auc": float(np.mean(valid_aucs)) if valid_aucs else None,  # Average ROC-AUC over classes with valid ROC curves.
        "macro_average_precision": float(np.mean(valid_aps)) if valid_aps else None,  # Average AP over classes for which AP was computed.
        "macro_f1": macro_f1,  # Return macro F1 without weighting by condition frequency.
        "micro_f1": float(f1_score(y_true.reshape(-1),  # Flatten all disease predictions and labels for global micro F1.
                                   predictions.reshape(-1), zero_division=0)),  # Complete the micro F1 calculation across all image-label pairs.
        "num_roc_auc_classes": len(valid_aucs),  # Show how many diseases contributed to the macro ROC-AUC.
        "per_class": per_class,  # Include detailed metrics for every individual condition.
    }  # Finish the dataset-level metrics dictionary.


def store_test_predictions(df, y_true, probs, out_path: Path):  # Save one row per test image with labels and model probabilities.
    result = df[["Image Index", "Patient ID"]].reset_index(drop=True).copy()  # Retain image and patient IDs for traceability.
    for i, condition in enumerate(CONDITIONS):  # Loop over 14 disease labels to add truth and probability columns.
        result[f"true_{condition}"] = y_true[:, i].astype(int)  # Record whether the disease actually appears in each image label.
        result[f"prob_{condition}"] = probs[:, i]  # Record the model's predicted probability for the disease.
    result.to_csv(out_path, index=False)  # Write the completed prediction table to a CSV file.

def fit_model(name, train_df, val_df, test_df, args, device):  # Train, select, evaluate, and save one of the requested models.
    print(f"\n{'=' * 20} {name.upper()} {'=' * 20}")  # Print a heading to distinguish model runs in the output log.
    model = build_model(name, args).to(device)  # Create this architecture and move its parameters to CPU/GPU/MPS.
    train_loader = get_loader(train_df, name, True, args, device)  # Build the training loader with random image augmentation.
    val_loader = get_loader(val_df, name, False, args, device)  # Build validation loader without random augmentation.
    test_loader = get_loader(test_df, name, False, args, device)  # Build test loader without random augmentation.

    train_targets = np.stack(train_df["targets"].to_numpy())  # Stack training-set targets to inspect label frequencies.
    if args.pos_weight:  # Check whether positive-class reweighting was requested.
        positive = train_targets.sum(axis=0)  # Count positive training images for each condition.
        negative = len(train_targets) - positive  # Count negative training images for each condition.
        # Missing labels in a small development sample get neutral weight.
        weights = np.where(positive > 0, negative / np.maximum(positive, 1), 1.0)  # Weight scarce positive examples by negative/positive frequency.
        loss_fn = nn.BCEWithLogitsLoss(  # Instantiate binary cross-entropy with optional positive weights.
            pos_weight=torch.tensor(weights, dtype=torch.float32, device=device))  # Send positive-class weights to the model's device.
    else:  # Use ordinary loss instead when class weighting is disabled.
        loss_fn = nn.BCEWithLogitsLoss()  # Create stable BCE-with-logits loss, which includes sigmoid internally.

    lr = args.lr_densenet if name == "densenet" else args.lr_baseline  # Select a smaller learning rate for DenseNet than for simple baselines.
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)  # Create Adam optimizer for parameters that are allowed to learn.
    num_params = sum(p.numel() for p in model.parameters())  # Count all model parameters, including any frozen parameters.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Count only parameters whose gradients are enabled.
    print(f"Parameters: {num_params:,} total, {trainable_params:,} trainable")  # Show model size and how many parameters are being optimized.

    best_score = float("-inf")  # Initialize the best validation selection score as negative infinity.
    best_state = None  # Reserve a variable for the best model parameter snapshot.
    best_epoch = 0  # Track the epoch that achieved the best validation score.
    epochs_without_improvement = 0  # Count consecutive epochs without improved validation performance.
    history = []  # Collect losses and validation ROC-AUC for plotting or reporting.

    for epoch in range(1, args.epochs + 1):  # Train for the requested maximum number of epochs.
        train_loss = train_one_epoch(  # Start the forward/backward pass over one training epoch.
            model, train_loader, loss_fn, optimizer, device,  # Provide the model, training batches, loss, optimizer, and device.
            freeze_backbone=(name == "densenet" and args.freeze_backbone)  # Freeze DenseNet feature layers when the corresponding flag is set.
        )  # Finish the training-epoch function call.
        val_loss, y_val, p_val = predict(model, val_loader, loss_fn, device)  # Evaluate loss and prediction probabilities on validation patients.
        # Threshold-independent ROC-AUC for best-epoch selection.
        val_metrics = compute_metrics(y_val, p_val,  # Compute class-wise validation metrics at temporary 0.5 cutoffs.
                                      np.full(NUM_CLASSES, 0.5, dtype=np.float64))  # Pass identical temporary cutoffs for all 14 diseases.
        val_auc = val_metrics["macro_roc_auc"]  # Read the macro validation ROC-AUC to rank candidate epochs.
        # In tiny datasets ROC-AUC may be undefined; fall back to -validation loss.
        selection_score = val_auc if val_auc is not None else -val_loss  # Use higher ROC-AUC to choose best epoch; lower loss if AUC undefined.
        history.append({"epoch": epoch, "train_loss": train_loss,  # Append epoch number, loss, and performance to the history.
                        "val_loss": val_loss, "val_macro_roc_auc": val_auc})  # Complete the history record with validation loss and macro AUC.
        auc_display = f"{val_auc:.4f}" if val_auc is not None else "undefined"  # Format optional ROC-AUC for readable printing.
        print(f"Epoch {epoch:02d}/{args.epochs}: train loss={train_loss:.4f}; "  # Print training loss and epoch index.
              f"val loss={val_loss:.4f}; val macro ROC-AUC={auc_display}")  # Finish the log line with validation loss and macro AUC.

        if selection_score > best_score + 1e-5:  # Check whether validation performance improved beyond tolerance.
            best_score = selection_score  # Update the best observed selection score.
            best_state = copy.deepcopy(model.state_dict())  # Deep-copy learned parameters so later epochs cannot overwrite them.
            best_epoch = epoch  # Record which epoch produced these weights.
            epochs_without_improvement = 0  # Reset early-stopping patience after an improvement.
        else:  # Handle epochs with no measurable validation improvement.
            epochs_without_improvement += 1  # Increase the consecutive no-improvement counter.
        if args.patience and epochs_without_improvement >= args.patience:  # Check whether early-stopping patience has been exhausted.
            print(f"Stopping early after {args.patience} epochs without improvement.")  # Explain why training stopped before the maximum epoch count.
            break  # Exit the training loop early to limit overfitting and wasted time.

    if best_state is None:  # Verify that at least one epoch produced a saved checkpoint.
        raise RuntimeError("No valid training epoch completed")  # Raise an error if the training loop produced no valid model.
    model.load_state_dict(best_state)  # Restore the parameters from the strongest validation epoch.
    model.eval()  # Set the restored model into evaluation mode.

    val_loss, y_val, p_val = predict(model, val_loader, loss_fn, device)  # Re-run validation on the restored best-performing checkpoint.
    thresholds = choose_thresholds(y_val, p_val)  # Tune disease thresholds using validation samples only.
    validation_metrics = compute_metrics(y_val, p_val, thresholds)  # Evaluate the final validation classification at tuned thresholds.
    validation_metrics["loss"] = float(val_loss)  # Attach validation loss to the output metrics dictionary.
    test_loss, y_test, p_test = predict(model, test_loader, loss_fn, device)  # Evaluate the selected model once on held-out test patients.
    test_metrics = compute_metrics(y_test, p_test, thresholds)  # Compute final test metrics using the unchanged validation thresholds.
    test_metrics["loss"] = float(test_loss)  # Attach test BCE loss to the results.

    model_dir = args.output_dir / name  # Choose a separate output folder for this architecture.
    model_dir.mkdir(parents=True, exist_ok=True)  # Create the folder, including any missing parent folders.
    torch.save({  # Write a checkpoint with model weights and interpretive metadata.
        "state_dict": model.state_dict(),  # Store all learned layer parameters and BatchNorm statistics.
        "model_name": name,  # Store the architecture identifier needed when reloading weights.
        "classes": CONDITIONS,  # Save the disease-label order associated with model outputs.
        "best_epoch": best_epoch,  # Save which epoch was selected by validation.
        "pretrained": (name == "densenet" and not args.no_pretrained),  # Record whether ImageNet initialization was requested.
    }, model_dir / "best_model.pt")  # Write the checkpoint into best_model.pt.
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)  # Save one row of training/validation statistics per epoch.
    store_test_predictions(test_df, y_test, p_test, model_dir / "test_predictions.csv")  # Save test labels and predicted probabilities in CSV format.

    summary = {  # Start the full model run summary dictionary.
        "model": name,  # Identify which model generated these results.
        "best_epoch": best_epoch,  # Record the checkpoint epoch used for testing.
        "total_parameters": num_params,  # Record all parameters in the model.
        "trainable_parameters": trainable_params,  # Record only the trainable subset of parameters.
        "train_images": len(train_df),  # Record the number of training X-rays.
        "val_images": len(val_df),  # Record the number of validation X-rays.
        "test_images": len(test_df),  # Record the number of test X-rays.
        "validation": validation_metrics,  # Attach all validation metrics, including per-disease results.
        "test": test_metrics,  # Attach all held-out test metrics, including per-disease results.
    }  # Finish constructing the run summary.
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:  # Open the JSON output file safely with UTF-8 encoding.
        json.dump(summary, f, indent=2, allow_nan=False)  # Serialize the complete summary using readable indentation.
    print(f"TEST macro ROC-AUC: {test_metrics['macro_roc_auc']}; "  # Print the final macro test ROC-AUC.
          f"macro AP: {test_metrics['macro_average_precision']}; "  # Print the macro test average precision.
          f"micro F1: {test_metrics['micro_f1']:.4f}")  # Print global micro F1 across all condition decisions.
    print(f"Outputs: {model_dir}")  # Show where the model outputs were saved.
    return summary  # Return this model's results for cross-model comparison.

def run_experiment(args):  # Run the same end-to-end experiment using notebook configuration instead of CLI arguments.
    if args.epochs < 1 or args.batch_size < 1:  # Reject zero or negative epoch counts and batch sizes.
        raise ValueError("epochs and batch-size must be positive")  # Explain that the training settings must be positive numbers.
    random.seed(args.seed)  # Seed Python random operations.
    np.random.seed(args.seed)  # Seed NumPy random operations.
    torch.manual_seed(args.seed)  # Seed CPU PyTorch random operations.
    if torch.cuda.is_available():  # Test whether an NVIDIA CUDA GPU is available.
        torch.cuda.manual_seed_all(args.seed)  # Seed all CUDA devices for better reproducibility.
        device = torch.device("cuda")  # Prefer CUDA acceleration where available.
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Otherwise check for Apple's Metal Performance Shaders accelerator.
        device = torch.device("mps")  # Use MPS when running on a supported Apple device.
    else:  # Handle computers without GPU accelerators.
        device = torch.device("cpu")  # Fall back to CPU computation.
    print(f"Device: {device}. Full DenseNet training is best run with a GPU.")  # Display the device selected for training.

    df = load_metadata(args.csv, args.images_dir)  # Read the NIH metadata and associate images with disease targets.
    train_df, val_df, test_df = split_by_patient(df, args.seed)  # Make a single patient-level split to reuse for every model.
    args.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the results folder exists.

    # Save split membership once, so every model sees exactly the same patients.
    parts = []  # Begin collecting split membership tables.
    for name, subset in (("train", train_df), ("val", val_df), ("test", test_df)):  # Visit the train, validation, and test subsets.
        parts.append(subset[["Image Index", "Patient ID"]].assign(split=name))  # Tag each image with its patient and split name.
    pd.concat(parts, ignore_index=True).to_csv(args.output_dir / "patient_splits.csv",  # Concatenate split tables and begin saving to CSV.
                                               index=False)  # Finish writing patient_splits.csv without DataFrame row indices.

    comparisons = []  # Prepare the list of model-comparison records.
    for name in dict.fromkeys(args.models):  # Loop over requested models while removing duplicate selections.
        metrics = fit_model(name, train_df, val_df, test_df, args, device)  # Train/evaluate this architecture using the shared split.
        test = metrics["test"]  # Extract the final held-out test metric dictionary.
        comparisons.append({  # Create one cross-model comparison row.
            "model": name,  # Save the model identifier in the row.
            "parameters": metrics["total_parameters"],  # Save parameter count for model-size comparisons.
            "best_epoch": metrics["best_epoch"],  # Save which epoch was selected for this run.
            "test_macro_roc_auc": test["macro_roc_auc"],  # Save macro test ROC-AUC for a threshold-free comparison.
            "test_macro_average_precision": test["macro_average_precision"],  # Save macro average precision for class-imbalance-sensitive comparison.
            "test_macro_f1": test["macro_f1"],  # Save mean per-disease F1.
            "test_micro_f1": test["micro_f1"],  # Save micro F1 pooled over all label decisions.
        })  # Finish this model-comparison row.
    comparison_df = pd.DataFrame(comparisons)  # Convert comparison records into a pandas table.
    comparison_df.to_csv(args.output_dir / "model_comparison.csv", index=False)  # Write the final comparison spreadsheet-like CSV file.
    print("\nMODEL COMPARISON (held-out test patients)")  # Announce the final results from held-out test patients.
    print(comparison_df.to_string(index=False))  # Print the readable model-comparison table.
    print(f"\nSaved results in {args.output_dir.resolve()}")  # Print the absolute path of the saved results directory.
    return comparison_df  # Also return the saved comparison table for display in Colab.

# Validate local paths before starting an expensive run.
if not args.csv.is_file():  # Check that the NIH metadata is visible to the local Python runtime.
    raise FileNotFoundError(  # Explain how to correct an inaccessible path.
        f'Metadata CSV not found: {args.csv}. Set DATA_ROOT/CSV_PATH to your actual local path and confirm that Colab is connected to your local runtime.'
    )  # Finish the actionable missing-CSV error.
if not args.images_dir.is_dir():  # Verify that the extracted image directory is visible too.
    raise FileNotFoundError(f'Image directory not found: {args.images_dir}')  # Ask the user to correct the folder path.

comparison_df = run_experiment(args)  # Train the requested models and evaluate on held-out patient groups.

from IPython.display import display  # Display pandas tables neatly in the notebook.
display(comparison_df)  # Show the held-out test metrics side by side.

"""## 11. Review your results

All results are saved directly to `OUTPUT_DIR` on your computer. They persist when Colab disconnects. The optional last cell creates a ZIP archive on the same drive; no browser upload or download is needed.
"""

from IPython.display import display  # Import display even when reviewing results in a fresh kernel.
# Load and show the model comparison table from disk (also useful after reconnecting).
comparison_csv = args.output_dir / 'model_comparison.csv'  # Locate the saved comparison results.
if comparison_csv.is_file():  # Check whether training produced the results file.
    display(pd.read_csv(comparison_csv))  # Render the summary in the notebook.
    print('Files saved:')  # Announce the results file inventory.
    for file_path in sorted(args.output_dir.rglob('*')):  # Recursively enumerate outputs for all trained models.
        if file_path.is_file():  # Only print actual files, not directories.
            print(' -', file_path.relative_to(args.output_dir))  # Print the file path relative to the results folder.
else:  # Handle an experiment that has not finished yet.
    print('No comparison table found yet. Run the training cell first.')  # Provide the next step.

import json
import pandas as pd
from IPython.display import display

# Safely load the completed SMALL_CNN metrics directly from its individual directory
cnn_metrics_path = args.output_dir / 'small_cnn' / 'metrics.json'

if cnn_metrics_path.is_file():
    with open(cnn_metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Restructure into a clean comparison format matching the final cells
    summary = [{
        "model": data["model"],
        "parameters": data["total_parameters"],
        "best_epoch": data["best_epoch"],
        "test_macro_roc_auc": data["test"]["macro_roc_auc"],
        "test_macro_average_precision": data["test"]["macro_average_precision"],
        "test_macro_f1": data["test"]["macro_f1"],
        "test_micro_f1": data["test"]["micro_f1"]
    }]

    print("\n--- Found SMALL_CNN results --- ")
    display(pd.DataFrame(summary))
else:
    print(f"Could not find the metrics file at: {cnn_metrics_path}")

import json
import pandas as pd
from IPython.display import display, HTML

# Define paths
cnn_dir = args.output_dir / 'small_cnn'
history_path = cnn_dir / 'training_history.csv'
metrics_path = cnn_dir / 'metrics.json'

# 1. Load and display training/validation history (Loss & ROC-AUC accuracy metric)
if history_path.is_file():
    print("\n=== Training & Validation Progress (Per Epoch) ===")
    history_df = pd.read_csv(history_path)
    # Rename columns for clarity
    history_df.columns = ['Epoch', 'Train Loss', 'Val Loss', 'Val Macro ROC-AUC (Accuracy)']
    display(history_df.style.hide(axis='index'))
else:
    print(f"Could not find training history at {history_path}")

# 2. Load and display final Testing Metrics
if metrics_path.is_file():
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)

    test_metrics = metrics_data.get('test', {})
    test_summary = {
        "Metric": [
            "Test Loss",
            "Test Macro ROC-AUC (Overall Accuracy)",
            "Test Macro Average Precision",
            "Test Macro F1-Score",
            "Test Micro F1-Score"
        ],
        "Value": [
            test_metrics.get("loss"),
            test_metrics.get("macro_roc_auc"),
            test_metrics.get("macro_average_precision"),
            test_metrics.get("macro_f1"),
            test_metrics.get("micro_f1")
        ]
    }
    print("\n=== Held-Out Test Dataset Performance ===")
    display(pd.DataFrame(test_summary).style.hide(axis='index'))
else:
    print(f"Could not find final metrics at {metrics_path}")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ensure we have our model loaded and in evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = args.output_dir / 'small_cnn' / 'best_model.pt'
predictions_path = args.output_dir / 'small_cnn' / 'test_predictions.csv'

# 1. Grad-CAM Class for SmallCNNBaseline
class SmallCNNGradCAM:
    def __init__(self, model):
        self.model = model
        self.eval_mode = model.eval()
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        # Hook into the last convolutional layer: features[6]
        target_layer = self.model.features[6]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Target score for the specific disease class
        score = output[0, class_idx]
        score.backward()

        # Average the gradients spatially to get weight alpha
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Linear combination of activations
        grad_cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to keep only features that positively contribute to the class score
        grad_cam = F.relu(grad_cam)

        # Normalize between 0 and 1
        grad_cam = grad_cam - grad_cam.min()
        grad_cam = grad_cam / (grad_cam.max() + 1e-8)
        return grad_cam.squeeze().detach().cpu().numpy()

# 2. Setup visualizer
if checkpoint_path.is_file() and predictions_path.is_file():
    # Initialize baseline and load weights
    model = SmallCNNBaseline().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # IMPORTANT: Disable in-place ReLU activations for Grad-CAM gradient calculation
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    # Instantiate Grad-CAM engine
    cam_extractor = SmallCNNGradCAM(model)

    # Load predictions
    preds_df = pd.read_csv(predictions_path)

    # Index the image directories to dynamically resolve raw image paths without needing the global df
    print("Indexing image directories to find the example files...")
    image_by_name = find_image_paths(args.images_dir)

    # We'll analyze 'Effusion' as it has a good number of positive targets
    target_class = "Effusion"
    class_idx = CONDITIONS.index(target_class)

    true_col = f"true_{target_class}"
    prob_col = f"prob_{target_class}"

    # Filter to cases where the condition is actually present
    pos_cases = preds_df[preds_df[true_col] == 1].copy()

    if not pos_cases.empty:
        # High Confidence Correct: Actual positive with highest probability score
        high_conf = pos_cases.sort_values(by=prob_col, ascending=False).iloc[0]

        # Low Confidence Correct: Actual positive with lowest probability score
        low_conf = pos_cases.sort_values(by=prob_col, ascending=True).iloc[0]

        examples = [
            ("High Confidence True Positive", high_conf),
            ("Low Confidence True Positive", low_conf)
        ]

        # Set up image transformations matching the original evaluation pipeline
        transform = make_transform("small_cnn", train=False)

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))

        for idx, (title, row) in enumerate(examples):
            img_idx = row["Image Index"]
            prob_val = row[prob_col]

            # Retrieve path from our newly built indexed directory map
            img_path = Path(image_by_name[img_idx])

            if img_path.is_file():
                # Load and preprocess image
                pil_img = Image.open(img_path).convert("L")
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                # Generate Grad-CAM activation map
                heatmap = cam_extractor.generate_heatmap(input_tensor, class_idx)

                # Suppress outermost 5% borders to eliminate frame artifacts / outside boundary lines
                h, w = heatmap.shape
                border_h, border_w = int(h * 0.05), int(w * 0.05)
                heatmap[:border_h, :] = 0
                heatmap[-border_h:, :] = 0
                heatmap[:, :border_w] = 0
                heatmap[:, -border_w:] = 0

                # Filter lower-intensity noise (threshold lowered to 0.12 to capture lung anatomy)
                heatmap = np.where(heatmap < 0.12, 0.0, heatmap)
                # Re-normalize contrast after thresholding to make the gradient values pop
                if heatmap.max() > 0:
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

                # Resize heatmap to match image dimension (224x224)
                resized_img = pil_img.resize((224, 224))
                resized_heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR)
                heatmap_np = np.array(resized_heatmap) / 255.0

                # Display original resized X-ray
                axes[idx, 0].imshow(resized_img, cmap='gray')
                axes[idx, 0].set_title(f"{title}\nImage: {img_idx}")
                axes[idx, 0].axis('off')

                # Display Grad-CAM Overlaid with 'turbo' colormap & high alpha
                axes[idx, 1].imshow(resized_img, cmap='gray')
                im_overlay = axes[idx, 1].imshow(heatmap_np, cmap='turbo', alpha=0.60, vmin=0.0, vmax=1.0)
                axes[idx, 1].set_title(f"Model Focus (Grad-CAM)\nPredicted Prob of {target_class}: {prob_val:.4f}")
                axes[idx, 1].axis('off')

                # Add colorbar next to the overlay image
                divider = make_axes_locatable(axes[idx, 1])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im_overlay, cax=cax, orientation="vertical")

        plt.tight_layout()
        plt.show()
    else:
        print(f"No positive cases found in test predictions for target class: {target_class}")
else:
    print("Ensure the small_cnn checkpoint and test_predictions are available in your output directory.")

# Optional: archive results on your hard drive without transferring any files.
import shutil  # Provide ZIP archive creation.
CREATE_RESULTS_ZIP = False  # Set True only if you want an extra copy of the saved results as a ZIP.

if CREATE_RESULTS_ZIP:  # Avoid extra disk usage unless the user requests an archive.
    if args.output_dir.is_dir() and any(args.output_dir.iterdir()):  # Only archive existing nonempty results.
        zip_path = args.output_dir.parent / (args.output_dir.name + '.zip')  # Save the archive beside the results folder.
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', root_dir=args.output_dir)  # Create the ZIP from local files.
        print('ZIP saved to:', zip_path.resolve())  # Show its absolute local path.
    else:  # Handle missing results gracefully.
        print('No results to archive. Run the training cell first.')  # Explain how to generate results.
else:  # Keep the original results without duplicating them.
    print('ZIP creation disabled. Results folder:', args.output_dir.resolve())  # Display the local results directory.
