from train import build_args
from importlib import import_module
from PIL import Image
import os

def denormalize_image(image):
    return (image * 0.5 + 0.5).clamp(0, 1)

def main():
    args = build_args()
    os.makedirs("generated", exist_ok=True)

    print("testing train_dataset")
    train_dataset = getattr(import_module("dataset"), "VITONHDDataset")(
        data_root_dir=args.data_root_dir,
        img_H=args.img_H,
        img_W=args.img_W,
        phase="train",
        transform_size=args.transform_size,
        transform_color=args.transform_color,
    )
    data = train_dataset[0]
    agn_pil = Image.fromarray(denormalize_image(data["agn"]).mul(255).byte().cpu().numpy())
    agn_mask_pil = Image.fromarray(data["agn_mask"].bool().squeeze(-1).cpu().numpy())
    hybvton_warped_cloth_pil = Image.fromarray(denormalize_image(data["hybvton_warped_cloth"]).mul(255).byte().cpu().numpy())
    hybvton_warped_mask_pil = Image.fromarray(data["hybvton_warped_mask"].bool().squeeze(-1).cpu().numpy())

    person_id = os.path.splitext(data["image_fn"])[0]
    cloth_id = os.path.splitext(data["cloth_fn"])[0]

    agn_pil.save(f"generated/{person_id}_{cloth_id}_agn.png")
    agn_mask_pil.save(f"generated/{person_id}_{cloth_id}_agn_mask.png")
    hybvton_warped_cloth_pil.save(f"generated/{person_id}_{cloth_id}_hybvton_warped_cloth.png")
    hybvton_warped_mask_pil.save(f"generated/{person_id}_{cloth_id}_hybvton_warped_mask.png")

    print("testing valid_paired_dataset")
    valid_paired_dataset = getattr(import_module("dataset"), "VITONHDDataset")(
        data_root_dir=args.data_root_dir,
        img_H=args.img_H,
        img_W=args.img_W,
        phase="val",
        is_paired=True,
        is_sorted=True,
    )
    data = valid_paired_dataset[0]
    agn_pil = Image.fromarray(denormalize_image(data["agn"]).mul(255).byte().cpu().numpy())
    agn_mask_pil = Image.fromarray(data["agn_mask"].bool().squeeze(-1).cpu().numpy())
    hybvton_warped_cloth_pil = Image.fromarray(
        denormalize_image(data["hybvton_warped_cloth"]).mul(255).byte().cpu().numpy())
    hybvton_warped_mask_pil = Image.fromarray(data["hybvton_warped_mask"].bool().squeeze(-1).cpu().numpy())

    person_id = os.path.splitext(data["image_fn"])[0]
    cloth_id = os.path.splitext(data["cloth_fn"])[0]

    agn_pil.save(f"generated/{person_id}_{cloth_id}_agn_val_paired.png")
    agn_mask_pil.save(f"generated/{person_id}_{cloth_id}_agn_mask_val_paired.png")
    hybvton_warped_cloth_pil.save(f"generated/{person_id}_{cloth_id}_hybvton_warped_cloth_val_paired.png")
    hybvton_warped_mask_pil.save(f"generated/{person_id}_{cloth_id}_hybvton_warped_mask_val_paired.png")

    print("testing valid_unpaired_dataset")
    valid_unpaired_dataset = getattr(import_module("dataset"), "VITONHDDataset")(
        data_root_dir=args.data_root_dir,
        img_H=args.img_H,
        img_W=args.img_W,
        phase="val",
        is_paired=False,
        is_sorted=True,
    )
    data = valid_unpaired_dataset[0]
    agn_pil = Image.fromarray(denormalize_image(data["agn"]).mul(255).byte().cpu().numpy())
    agn_mask_pil = Image.fromarray(data["agn_mask"].bool().squeeze(-1).cpu().numpy())
    hybvton_warped_cloth_pil = Image.fromarray(
        denormalize_image(data["hybvton_warped_cloth"]).mul(255).byte().cpu().numpy())
    hybvton_warped_mask_pil = Image.fromarray(data["hybvton_warped_mask"].bool().squeeze(-1).cpu().numpy())

    person_id = os.path.splitext(data["image_fn"])[0]
    cloth_id = os.path.splitext(data["cloth_fn"])[0]

    agn_pil.save(f"generated/{person_id}_{cloth_id}_agn_val_unpaired.png")
    agn_mask_pil.save(f"generated/{person_id}_{cloth_id}_agn_mask_val_unpaired.png")
    hybvton_warped_cloth_pil.save(f"generated/{person_id}_{cloth_id}_hybvton_warped_cloth_val_unpaired.png")
    hybvton_warped_mask_pil.save(f"generated/{person_id}_{cloth_id}_hybvton_warped_mask_val_unpaired.png")

    print("done")
if __name__ == "__main__":
    main()
