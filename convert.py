import json
import os

def convert_coco_to_yolo(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}

    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_annotations.setdefault(img_id, []).append(ann)

    for img_id, anns in img_annotations.items():
        img_info = images[img_id]
        img_w, img_h = img_info['width'], img_info['height']
        
        txt_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'w') as out_f:
            for ann in anns:
                # COCO bbox: [x_min, y_min, width, height]
                x_min, y_min, w, h = ann['bbox']
                category_id = ann['category_id'] - 1  # 0-indexed class

                # Normalize values for YOLO [class_id, x_center, y_center, width, height]
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                out_f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

convert_coco_to_yolo("seadronessee/annotations/instances_train.json", "seadronessee/labels/train")
convert_coco_to_yolo("seadronessee/annotations/instances_val.json", "seadronessee/labels/val")
