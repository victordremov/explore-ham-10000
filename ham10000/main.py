import os
import sys
from collections import defaultdict
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from ignite.engine import (
    Engine,
    create_supervised_trainer,
    create_supervised_evaluator,
    Events,
)
from ignite.metrics import Loss, Accuracy
from numpy.core.multiarray import ndarray
from typing import Dict, Tuple, List, Optional, Callable, DefaultDict

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

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


MODEL_PATH = "best_model.pth"


def preprocess_metadata(images_directory: str, metadata_filename: str) -> MetaData:
    original_metadata = pd.read_csv(metadata_filename)
    original_metadata["diagnosis_idx"] = encode_diagnoses(original_metadata)
    add_path_column_inplace(images_directory, original_metadata)
    return original_metadata


def encode_diagnoses(metadata: MetaData) -> ndarray:
    return sklearn.preprocessing.LabelEncoder().fit_transform(metadata["dx"])


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
        GroupShuffleSplit(n_splits=1, test_size=holdout_share)
        .split(X=metadata, groups=metadata["lesion_id"])
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
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.metadata: pd.DataFrame = metadata.reset_index(drop=True)
        if transform is None:
            transform = Compose([ToTensor()])
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.transform(Image.open(self.metadata.at[index, "path"]))
        y = torch.tensor(int(self.metadata.at[index, "diagnosis_idx"]))
        return X, y


def initialize(model: nn.Module, n_classes: int, device: torch.device):
    class ImageOnlyModel(nn.Module):
        def __init__(self, model: nn.Module, n_classes: int):
            super().__init__()
            self.cnn = model
            for parameter in self.cnn.parameters():
                parameter.requires_grad = False

            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, n_classes)

        def forward(self, image):
            x = self.cnn(image)
            return x

    image_only_model = ImageOnlyModel(model, n_classes=n_classes)
    image_only_model.to(device, non_blocking=True)
    return image_only_model


def main() -> Tuple[
    nn.Module, DefaultDict[str, List[float]], DefaultDict[str, List[float]]
]:
    directory = Path(__file__) / ".."

    metadata = preprocess_metadata(
        images_directory=directory / "images",
        metadata_filename=directory / "HAM10000_metadata.csv",
    )

    metadata_train, metadata_validation = split_train_validation(
        metadata, holdout_share=0.1
    )
    transform = transforms.Compose(
        [
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_loader = DataLoader(
        dataset=HAMDataset(metadata_train, transform=transform),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    validation_loader = DataLoader(
        dataset=HAMDataset(metadata_validation, transform=transform),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize(models.resnet152(pretrained=True), n_classes=7, device=device)

    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)

    loss = nn.CrossEntropyLoss().to(device)
    trainer = create_supervised_trainer(
        model=model, optimizer=optimizer, loss_fn=loss, device=device
    )
    evaluator = create_supervised_evaluator(
        model=model,
        metrics={"cross-entropy": Loss(loss), "accuracy": Accuracy()},
        device=device,
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer: Engine) -> None:
        print(f".", end="", file=sys.stderr)

    train_results: DefaultDict[str, List[float]] = defaultdict(list)
    validation_results: DefaultDict[str, List[float]] = defaultdict(list)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer: Engine) -> None:
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['cross-entropy']:.2f}"
        )
        for metric_name, metric_value in metrics.items():
            train_results[metric_name].append(metric_value)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer: Engine) -> None:
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['cross-entropy']:.2f}"
        )
        for metric_name, metric_value in metrics.items():
            validation_results[metric_name].append(metric_value)

    scheduller = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.2, patience=3, verbose=True
    )

    @evaluator.on(Events.COMPLETED)
    def reduce_learning_rate_on_plateau(evaluator: Engine) -> None:
        scheduller.step(metrics=evaluator.state.metrics["cross-entropy"])

    best_validation_loss = None

    @evaluator.on(Events.COMPLETED)
    def store_best_model(_: Engine) -> None:
        nonlocal best_validation_loss
        validation_loss = evaluator.state.metrics["cross-entropy"]
        if best_validation_loss is None or validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), MODEL_PATH)

    trainer.run(train_loader, max_epochs=10)
    return model, train_results, validation_results


if __name__ == "__main__":
    model, train_metrics, validation_metrics = main()
    print(
        """\
        Train metrics:
        {metrics}
    """.format(
            metrics="\n".join(
                f"{name}: {scores}" for name, scores in train_metrics.items()
            )
        )
    )
    print(
        """\
        Validation metrics:
        {metrics}
    """.format(
            metrics="\n".join(
                f"{name}: {scores}" for name, scores in train_metrics.items()
            )
        )
    )
