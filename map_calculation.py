import os
from PIL import Image
import torch
import detect_mod
from pprint import pprint
import pandas as pd
import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def save_maps(data_dict):
    mod_data = []
    random_data = []

    for path, values in data_dict.items():
        # Convert the tensors to standard python data types
        map_val = values['map'].item()
        map_50_val = values['map_50'].item()
        map_75_val = values['map_75'].item()

        # Construct a dictionary for the row
        row = {'map': map_val, 'map_50': map_50_val, 'map_75': map_75_val}

        if 'mod' in path or 'original' in path:
            # Extract the ratio from the path, or use NaN if it's the 'original' path
            ratio = float(path.split('/')[-2]) if 'mod' in path else np.nan
            row['ratio'] = ratio
            mod_data.append(row)
        elif 'random' in path:
            # Extract the ratio from the path
            ratio = float(path.split('/')[-2])
            row['ratio'] = ratio
            random_data.append(row)

    # Create the DataFrames
    mod_df = pd.DataFrame(mod_data)
    random_df = pd.DataFrame(random_data)

    # Merge the two DataFrames on 'ratio'
    merged_df = pd.merge(mod_df, random_df, on='ratio', suffixes=('_mod', '_random'), how='outer')
    merged_df = merged_df.sort_values('ratio')
    # Save the DataFrame to an Excel file
    merged_df.to_excel('output.xlsx', index=False)

    # Show the merged DataFrame
    print(merged_df)


def load_data(image_dir, label_dir):
    # Sorted list of image and label files
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    images = []
    targets = []

    for image_file, label_file in zip(image_files, label_files):
        # Ensure corresponding pairs
        assert image_file.split('/')[-1][:-4] == label_file.split('/')[-1][:-4]

        # Load image
        image = Image.open(os.path.join(image_dir, image_file))
        image_width, image_height = image.size
        # image = torch.Tensor(image)  # Convert to tensor
        images.append(image)

        # Load labels
        with open(os.path.join(label_dir, label_file), 'r') as f:
            labels = f.readlines()
            boxes = []
            classes = []
            for label in labels:
                cls, x, y, w, h = map(float, label.split())
                # Convert normalized coordinates to original size
                x = x * image_width
                y = y * image_height
                w = w * image_width
                h = h * image_height
                # Convert to x1, y1, x2, y2 format
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                boxes.append([x1, y1, x2, y2])
                classes.append(int(cls))

            targets.append({
                    "boxes": torch.tensor(boxes),
                    "labels": torch.tensor(classes)
                })
    
    return images, targets


def prepare_preds(detections):
    keys = sorted([ int(item) for item in detections.keys()])
    preds = []

    for key in keys:
        boxes = []
        classes = []
        scores = []

        for item in detections[str(key)]:
            cls, x1, y1, x2, y2, conf = item
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
            scores.append(conf)

        preds.append({
            "boxes": torch.tensor(boxes),
            "labels": torch.tensor(classes),
            "scores": torch.tensor(scores)
        })
    
    return preds

def calculate_map(image_path):
    # Load your data
    print(image_path)
    images, targets = load_data(image_path, "./runs/experiment/original/label/")
    detections = detect_mod.run_detection(detection_model, images, **detection_config)
    preds = prepare_preds(detections)

    metric = MeanAveragePrecision()
    metric.update(preds, targets)
    return metric.compute()


if __name__=="__main__":
    detection_config = vars(detect_mod.parse_opt())
    detection_model = detect_mod.warmup_model(weights=detection_config['weights'],
                                                data=detection_config['data'])
    for key in ['weights', 'data', 'imgsz', 'device', 'half', 'dnn', 'source']:
        detection_config.pop(key)

    image_path = "./runs/experiment/original/image/"

    maps = {image_path: calculate_map(image_path)} 

    base_path = f"./runs/experiment/mod/"

    for subdir in os.listdir(base_path):
        image_path = f"{base_path}{subdir}/"

        maps.get(image_path, {})
        maps[image_path] = calculate_map(image_path)

    base_path = f"./runs/experiment/random/"

    for subdir in os.listdir(base_path):
        image_path = f"{base_path}{subdir}/"

        maps.get(image_path, {})
        maps[image_path] = calculate_map(image_path)

    save_maps(maps)

    # keys = sorted([ item for item in maps.keys()])

    # for key in keys:
    #     print(key)
    #     pprint(maps[key])