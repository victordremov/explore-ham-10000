import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from dataclasses import dataclass
from ignite.engine import (
    Engine,
    create_supervised_trainer,
    create_supervised_evaluator,
    Events,
)
from ignite.metrics import Loss, Accuracy, ConfusionMatrix
from numpy.core.multiarray import ndarray
from typing import Dict, Tuple, List, Optional, Callable, DefaultDict

from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose, ToTensor

from ham10000.type_bindings import ImageId, TrainWithValidation, Holdout, MetaData

import os
from glob import glob
from PIL import Image

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


MODEL_PATH = "best_model.pth"


def preprocess_metadata(
    images_directory: Path, metadata_filename: Path
) -> Tuple[MetaData, sklearn.preprocessing.LabelEncoder, List[str]]:
    original_metadata = pd.read_csv(metadata_filename, na_values=["unknown"])
    original_metadata["diagnosis_idx"], class_encoder = encode_diagnoses(
        original_metadata
    )
    add_path_column_inplace(str(images_directory), original_metadata)
    paths = original_metadata.groupby(by="lesion_id").agg(
        {"path": lambda x: x.tolist()}
    )
    paths.rename(columns={"path": "paths"}, inplace=True)
    metadata = original_metadata.groupby(by="lesion_id")[
        ["diagnosis_idx", "lesion_id", "sex", "localization", "age"]
    ].first()
    metadata = metadata.join(paths)
    metadata.reset_index(inplace=True, drop=True)
    categorical_features_columns = []
    for column in ["sex", "localization"]:
        known_values = metadata.loc[~metadata[column].isnull(), column]
        one_hot_encoded = pd.get_dummies(known_values)
        one_hot_encoded_columns = one_hot_encoded.columns.tolist()
        categorical_features_columns.extend(one_hot_encoded_columns)
        metadata = metadata.join(one_hot_encoded)
        metadata.fillna(
            metadata[one_hot_encoded_columns].mean().to_dict(), inplace=True
        )
    age_groups = np.unique(metadata.loc[~metadata["age"].isnull(), "age"]).astype(int)
    age_columns = []
    for age_threshold in age_groups[1:]:
        column = f"age_at_least_{age_threshold}"
        metadata[column] = (metadata["age"] >= age_threshold)
        age_columns.append(column)
    categorical_features_columns.extend(age_columns)
    metadata.fillna(metadata[age_columns].mean().to_dict())
    metadata[categorical_features_columns] = metadata[categorical_features_columns].astype(np.float32)
    return metadata, class_encoder, categorical_features_columns


def encode_diagnoses(
    metadata: MetaData
) -> Tuple[ndarray, sklearn.preprocessing.LabelEncoder]:
    diagnosis_encoder = sklearn.preprocessing.LabelEncoder()
    diagnosis_encoder.fit(metadata["dx"])
    return diagnosis_encoder.transform(metadata["dx"]), diagnosis_encoder


def add_path_column_inplace(images_directory: str, metadata: MetaData) -> None:
    all_image_path = glob(os.path.join(images_directory, "*.jpg"))
    image_id_to_path: Dict[ImageId, Path] = {
        os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path
    }
    metadata["path"] = metadata["image_id"].map(image_id_to_path)


def split_train_validation(
    metadata: MetaData, holdout_share: float
) -> Tuple[TrainWithValidation, Holdout]:
    train_indices: ndarray
    validation_indices: ndarray
    train_indices, validation_indices = (
        StratifiedShuffleSplit(n_splits=1, test_size=holdout_share)
        .split(X=metadata, y=metadata["diagnosis_idx"])
        .__iter__()
        .__next__()
    )
    metadata_train = metadata.iloc[train_indices, :]
    metadata_validation = metadata.iloc[validation_indices, :]
    assert (
        len(
            np.intersect1d(
                metadata_train["lesion_id"], metadata_validation["lesion_id"]
            )
        )
        == 0
    )
    return metadata_train, metadata_validation


class HAMDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        categorical_features_columns: List[str],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.metadata: pd.DataFrame = metadata.reset_index(drop=True)
        if transform is None:
            transform = Compose([ToTensor()])
        self.transform = transform
        self.categorical_features = metadata.loc[:, categorical_features_columns].values

    def __len__(self):
        return len(self.metadata)

    def __getitem__(
        self, index: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        paths_to_images = self.metadata.at[index, "paths"]
        path_to_random_image = random.choice(paths_to_images)
        image_as_tensor = self.transform(Image.open(path_to_random_image))
        labels = torch.tensor(int(self.metadata.at[index, "diagnosis_idx"]))
        categorical_features = torch.tensor(self.categorical_features[index, :])
        return (image_as_tensor, categorical_features), labels


def initialize(
    model: nn.Module, n_classes: int, extra_in_features: int, device: torch.device
):
    class ImageWithCategoricalFeaturesModel(nn.Module):
        def __init__(self, model: nn.Module, n_classes: int, extra_in_features: int):
            super().__init__()
            self.cnn = model
            total_in_features = model.classifier.in_features + extra_in_features
            self.cnn.classifier = nn.Identity()
            self.last_fully_connected_layer = nn.Linear(
                in_features=total_in_features, out_features=n_classes
            )

        def forward(self, image_and_categorical_features):
            image, categorical_features = image_and_categorical_features
            image_features = self.cnn(image)
            features_united = torch.cat([image_features, categorical_features], dim=1)
            log_probabilities = self.last_fully_connected_layer(features_united)
            return log_probabilities

    wrapper = ImageWithCategoricalFeaturesModel(
        model, n_classes=n_classes, extra_in_features=extra_in_features
    )
    wrapper.to(device, non_blocking=True)
    return wrapper


@dataclass
class ConfusionMatrixDTO:
    matrix_data: Optional[ndarray]
    class_labels: List[str]


def main(
    max_epochs: int
) -> Tuple[
    nn.Module,
    DefaultDict[str, List[float]],
    DefaultDict[str, List[float]],
    ConfusionMatrixDTO,
]:
    directory = Path(__file__) / ".."

    metadata, class_encoder, categorical_features_columns = preprocess_metadata(
        images_directory=directory / "images",
        metadata_filename=directory / "HAM10000_metadata.csv",
    )

    metadata_train, metadata_validation = split_train_validation(
        metadata, holdout_share=0.1
    )
    transform = transforms.Compose(
        [
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    num_workers = 1
    batch_size = 16
    train_loader = DataLoader(
        dataset=HAMDataset(
            metadata=metadata_train,
            transform=transform,
            categorical_features_columns=categorical_features_columns,
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        dataset=HAMDataset(
            metadata_validation,
            transform=transform,
            categorical_features_columns=categorical_features_columns,
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize(
        models.densenet121(pretrained=True),
        n_classes=7,
        extra_in_features=len(categorical_features_columns),
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss = nn.CrossEntropyLoss().to(device)
    trainer = create_supervised_trainer(
        model=model, optimizer=optimizer, loss_fn=loss, device=device
    )
    evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            "cross-entropy": Loss(loss),
            "confusion-matrix": ConfusionMatrix(num_classes=7),
        },
        device=device,
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(_: Engine) -> None:
        print(f".", end="", file=sys.stderr)

    train_results: DefaultDict[str, List[float]] = defaultdict(list)
    validation_results: DefaultDict[str, List[float]] = defaultdict(list)

    best_validation_loss = None
    confusion_matrix = ConfusionMatrixDTO(
        class_labels=class_encoder.classes_.tolist(), matrix_data=None
    )

    def store_best_model(engine: Engine) -> None:
        nonlocal best_validation_loss
        nonlocal confusion_matrix
        metrics = engine.state.metrics
        validation_loss = metrics["cross-entropy"]
        if best_validation_loss is None or validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            # torch.save(model.state_dict(), MODEL_PATH)
            confusion_matrix.matrix_data = metrics["confusion-matrix"].numpy()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer: Engine) -> None:
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {trainer.state.epoch} Avg loss: {metrics['cross-entropy']:.2f}"
        )
        for metric_name, metric_value in metrics.items():
            train_results[metric_name].append(metric_value)

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.2, patience=3, verbose=True
    )

    def reduce_learning_rate_on_plateau(engine: Engine) -> None:
        scheduler.step(metrics=engine.state.metrics["cross-entropy"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine: Engine) -> None:
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {engine.state.epoch} Avg loss: {metrics['cross-entropy']:.2f}"
        )
        for metric_name, metric_value in metrics.items():
            validation_results[metric_name].append(metric_value)
        reduce_learning_rate_on_plateau(evaluator)
        store_best_model(evaluator)

    trainer.run(train_loader, max_epochs=max_epochs)
    return model, train_results, validation_results, confusion_matrix


if __name__ == "__main__":
    torch.cuda.empty_cache()
    _, train_metrics, validation_metrics, confusion_matrix = main(10)
    print(f"Train metrics: {train_metrics}.")
    print(f"Validation metrics: {validation_metrics}.")
    print(f"Confusion matrix: {confusion_matrix}.")
