import os
import cv2
import torch
from torchvision import transforms

def load_images(path, target_size=(40, 40)):
    loaded_images = []

    # Get the list of files sorted by modification time
    sorted_files = sorted(os.listdir(path), key=lambda x: os.path.getmtime(os.path.join(path, x)))

    for filename in sorted_files:
        img_path = os.path.join(path, filename)

        if os.path.isfile(img_path) and (filename.lower().endswith('.png') or filename.lower().endswith('.bmp')):
            # Read the image using OpenCV
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            binary_img = cv2.resize(img_array, target_size)



            # Binarize the image based on the threshold
            _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

            # Resize the binary image to the target size


            # Convert the binary image array to a PyTorch tensor

            img_tensor = transforms.ToTensor()(binary_img)




            loaded_images.append(img_tensor)

    # Stack the loaded tensors
    return torch.stack(loaded_images)
