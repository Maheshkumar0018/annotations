# input: ./train
#            - image_1.jpg,
#            - image_1.txt,
#            - image_2.jpg,
#            - image_2.txt

import os
import xml.etree.ElementTree as ET

class_labels = ['ambulance', 'auto-rickshaw', 'bicycle', 'bus', 'car', 'garbage van', 'human hauler', 'minibus',
                'minivan', 'motorbike', 'Pickup', 'army vehicle', 'police car', 'rickshaw', 'scooter', 'Suv', 'taxi',
                'three-wheelers (CNG)', 'truck', 'van', 'wheelbarrow'] # nc: 21

# Function to convert XML annotations to YOLO format
def convert_to_yolo_format(xml_file, class_labels):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_labels:
            continue

        class_id = class_labels.index(class_name)
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # YOLO format requires normalized coordinates
        x_center = (xmin + xmax) / (2.0 * width)
        y_center = (ymin + ymax) / (2.0 * height)
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_lines.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

    return yolo_lines

data_dir = "./train"

output_dir = "./annotations"

# Iterate over XML files and convert them to YOLO format
for xml_file in os.listdir(data_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(data_dir, xml_file)
        yolo_lines = convert_to_yolo_format(xml_path, class_labels)
        txt_filename = os.path.splitext(xml_file)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, "w") as txt_file:
            txt_file.write("\n".join(yolo_lines))

print("Conversion completed.")
