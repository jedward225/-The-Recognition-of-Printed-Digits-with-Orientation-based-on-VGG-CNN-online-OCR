from PIL import Image
import os

def combine_images_into_grid(image_files, grid_size=(10, 10), image_size=(128, 128)):
    # 创建一个新的空白图像，用于放置所有的小图片
    combined_image = Image.new('RGB', (grid_size[0] * image_size[0], grid_size[1] * image_size[1]))

    for index, file in enumerate(image_files):
        row = index // grid_size[0]
        col = index % grid_size[0]

        img = Image.open(file)
        
        combined_image.paste(img, (col * image_size[0], row * image_size[1]))
    
    return combined_image

# 获取所有图片文件名
image_folder = 'trainpic\\3'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

# 检查图片数量是否正确
if len(image_files) != 100:
    raise ValueError("Expected 100 images, found {}.".format(len(image_files)))

# 合并图片
combined_image = combine_images_into_grid(image_files)

# 保存合并后的图片
combined_image.save('combined_image.png')