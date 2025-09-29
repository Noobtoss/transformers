import os, datetime
import csv
import argparse
import torch
from torch import nn
from PIL import Image
from datasets import Dataset
import evaluate
from transformers import (
    TrainerCallback,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer
)


def make_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="YOLOX ONNX Deployment")
    parser.add_argument("--data_dir",
                        type=str,
                        default="/Users/noobtoss/code-nexus/transformers/datasets/holz00"
                        )
    return parser


def main():
    args = make_parser().parse_args()
    data_dir = args.data_dir
    NUM_CLASSES = 6  # Replace with your number of classes
    RUN_NAME = f"segformer_{os.path.basename(os.path.normpath(data_dir))}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(RUN_NAME)
    OUT_DIR = f"../runs/{RUN_NAME}"

    # ------------------------------
    # 1. Load ADE20K-style dataset
    # ------------------------------
    def load_ade20k_dataset(image_dir, mask_dir):
        images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        dataset_dict = {"image": images, "mask": masks}
        return Dataset.from_dict(dataset_dict)

    train_ds = load_ade20k_dataset(f"{data_dir}/train/images", f"{data_dir}/train/masks")
    val_ds = load_ade20k_dataset(f"{data_dir}/test/images", f"{data_dir}/test/masks")

    # Split into train/validation
    # dataset = dataset.train_test_split(test_size=0.2)
    # train_ds = dataset["train"]
    # val_ds = dataset["test"]

    print(train_ds)
    print(val_ds)

    # ------------------------------
    # 2. Preprocessing
    # ------------------------------
    image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    def preprocess(batch):
        images = [Image.open(img).convert("RGB") for img in batch["image"]]
        labels = [Image.open(mask).convert("L") for mask in batch["mask"]]
        encoded = image_processor(images, labels)
        return encoded

    train_ds.set_transform(preprocess)  #  train_ds = train_ds.map(preprocess)
    val_ds.set_transform(preprocess)  # val_ds = val_ds.map(preprocess)

    # ------------------------------
    # 3. Load SegFormer model
    # ------------------------------

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True  # <--- this fixes your error
    )

    # ------------------------------
    # 4. Define metric
    # ------------------------------

    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=NUM_CLASSES,
                ignore_index=0,
                reduce_labels=image_processor.do_reduce_labels,
            )

            # add per category metrics as individual key-value pairs
            per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            metrics.update({f"accuracy_{i}": v for i, v in enumerate(per_category_accuracy)})
            metrics.update({f"iou_{i}": v for i, v in enumerate(per_category_iou)})
            return metrics

    class SaveMetricsCallback(TrainerCallback):
        def __init__(self, save_path=f"{OUT_DIR}/metrics.csv"):
            self.save_path = save_path
            self.header_written = False
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                # Ensure epoch comes first
                if "epoch" in metrics:
                    metrics = {"epoch": f"{int(metrics.pop('epoch')):02d}", **metrics}

                with open(self.save_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=metrics.keys())

                    if not self.header_written:
                        writer.writeheader()
                        self.header_written = True

                    writer.writerow(metrics)

    # ------------------------------
    # 5. Training
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        save_steps=500,
        eval_steps=500,
        logging_steps=200,
        weight_decay=0.01,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        callbacks=[SaveMetricsCallback()]  # <--- here
    )

    trainer.train()


if __name__ == "__main__":
    main()
