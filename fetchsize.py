<<<<<<< HEAD
from PIL import Image
import os

data_dir = r'C:\Users\22597\Documents\GitHub\boiling\trainpic'
sizes = []

for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for img_path in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_path)
        if img_path.endswith('.png'):
            img = Image.open(img_path)
            sizes.append(img.size)

sizes.sort()
print("Image size distribution:")
for size in sizes:
=======
from PIL import Image
import os

data_dir = r'C:\Users\22597\Documents\GitHub\boiling\trainpic'
sizes = []

for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for img_path in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_path)
        if img_path.endswith('.png'):
            img = Image.open(img_path)
            sizes.append(img.size)

sizes.sort()
print("Image size distribution:")
for size in sizes:
>>>>>>> 81c37fa34b06a5f06fc0dd78f7c45157432b6766
    print(size)