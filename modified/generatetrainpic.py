# char_img_gen.py 生成训练样本

import char_img_gen as cg

X, Y = cg.gen_batch_examples(batch_size = 100, img_size = 32, font_dir = './font', output_path = 'C:\\Users\\22597\\Documents\\GitHub\\boiling\\modified\\example')


#先导入所需的包
import pygame
import os
import string
import random
import shutil  
import cv2
import numpy as np

pygame.init()  #  初始化

def generate_random_str(geshu):#生成不规则字符串
    stringlist=list()
    for i in range(geshu):
        random_str = ''
        base_str = '-().%0123456789'
        length = len(base_str) - 1
        for i in range(random.randint(3,10)):
            random_str += base_str[random.randint(0, length)]
        # print(random_str)
        stringlist.append(random_str)
    return stringlist;
def get_correct(geshu):#生成随机两位浮点数
    stringlist=list()
    for i in range(geshu):
        j=random.randint(1,9)
        maxnum=pow(10,j)
        num=round(random.uniform(-maxnum, maxnum),2)
        # print("{:,}".format(num))
        stringlist.append("{:,}".format(num))
    return stringlist;


def rotate_bound(image, angle):
    #获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
  
    # 提取旋转矩阵 sin cos 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
  
    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
    nH = h
  
    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
  
    return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

try:
	shutil.rmtree('image')#强制删除
except:
	pass
try:
	os.mkdir("image") #创建目录
except:
	pass
try:
	os.remove("label.txt")
except:
	pass
randomcount=eval(input("请输入生成的随机浮点数字符串数量："))
numbercount=eval(input("请输入生成的随机符号字符串数量："))
rotatedmaxangel=eval(input("请输入最大随机旋转角度："))
randomstringlist = get_correct(randomcount)#在括号里输入生成数量
numberstringlist = generate_random_str(numbercount)#在括号里输入生成数量

list = numberstringlist+randomstringlist#列表合并
listcount=randomcount+numbercount
#设置字体大小及路径
font = pygame.font.Font(os.path.join("C:\\Windows\\Fonts\\STKAITI.TTF"), 26)

with open('label.txt','a') as file_handle:   # .txt可以不自己新建,代码会自动新建  
	for i in range(listcount):
		text = u"{0}".format(list[i])           #  引号内引用变量使用字符串格式化
		#设置位置及颜色
		rtext = font.render(text, True, (0, 0, 0), (255 ,255 ,255))
		#保存图片及路径
		path=".\\image\\"+str(i)+".png"
		pygame.image.save(rtext, path)
		image = cv2.imread(path)#再次读取
		rotated = rotate_bound(image, random.randint(-rotatedmaxangel,rotatedmaxangel))#转转转
		cv2.imwrite(path, rotated)#保存旋转的图片
		file_handle.write("image/"+str(i)+".png"+"\t"+list[i]+"\n")#写入标签文件

print("完成，请查看目录下内容")
