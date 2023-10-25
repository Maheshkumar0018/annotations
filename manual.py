import cv2
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('./runs/content/runs/detect/train/weights/last.pt')

# Prompt the user to enter the image path
image_path = input("Enter the path to the image: ")

# Check if the entered image path is valid
if not os.path.isfile(image_path):
    print("Invalid image path. Please provide a valid path to an image.")
else:
    # Perform object detection
    results = model(image_path)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = results[0]

    if result is not None and len(result) > 0:
        object_counter = 1
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0][:4] 
            conf = box.conf
            cls = box.cls
            label = model.names[int(cls)] 
            print(f"---id: {object_counter}---: {label}")
            print("***Detected Object***:", label)
            print("***Coordinates***:", (x1, y1, x2, y2))
            print("***Probability***:", conf)
            text = f'{label}: {conf.item():.2f}'
            
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            object_counter += 1

        cv2.imshow('Frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_path = './detection/output_image.jpg'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Output image saved to:", output_path)

    else:
        print("No results.")
