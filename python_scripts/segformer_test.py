import argparse
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


def make_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="YOLOX ONNX Deployment")
    parser.add_argument("--ckpt_dir",
                        type=str,
                        default="/Users/noobtoss/code-nexus/transformers/checkpoints/segformer_holz"
                        )
    parser.add_argument("--img_path",
                        type=str,
                        default="/Users/noobtoss/code-nexus/transformers/datasets/holz00/test/images/F1150641.jpg"

                        )
    return parser

def main():
    HOLZ_COLORS = np.array(
        [[0, 0, 0], [0, 255, 0], [255, 255, 0], [0, 0, 255], [255, 255, 255], [255, 0, 0], [255, 165, 0]],
        dtype=np.uint8
    )

    args = make_parser().parse_args()
    ckpt_dir = args.ckpt_dir
    img_path = args.img_path

    # reload model + processor
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt_dir)
    processor = SegformerImageProcessor.from_pretrained(ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # load a test image
    image = Image.open(img_path).convert("RGB")

    # preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # shape [batch, num_classes, H/4, W/4]

    # upsample logits to match original image size
    logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )

    # get predicted class per pixel
    pred_mask = logits.argmax(dim=1)[0].cpu().numpy()  # shape (H, W)

    color_mask = HOLZ_COLORS[pred_mask]  # map classes to colors
    color_mask_img = Image.fromarray(color_mask).resize(image.size)

    alpha = 0.60
    overlay = Image.blend(image.convert("RGB"), color_mask_img, 1-alpha)
    overlay.show()

if __name__ == "__main__":
    main()
