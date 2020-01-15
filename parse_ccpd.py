import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np

'''
参考：
   https://blog.csdn.net/yang_daxia/article/details/88234138
   https://github.com/detectRecog/CCPD

例子：
    0073-17_3-360&460_461&521-448&521_360&494_373&460_461&487-0_0_11_25_21_33_29-113-17.jpg
    0078-15_11-480&489_566&565-566&542_484&565_480&512_562&489-0_0_10_4_29_25_30-55-8.jpg

规则：由分隔符'-'分为几个部分:
    1)025                               为车牌占整图的面积，感觉没啥用
    2)95_113                            对应两个角度, 水平95°, 竖直113°
    3)154&383_386&473                   对应边界框坐标:左上(154, 383), 右下(386, 473)
    4)386&473_177&454_154&383_363&402   对应四个角点坐标
    5)0_0_22_27_27_33_16                为车牌号码 映射关系如下: 第一个为省份0 对应省份字典皖, 后面的为字母和文字, 查看ads字典.如0为A, 22为Y.....
            provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
            ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                    'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    6)37                                亮度
    7)15                                模糊度
'''
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

f_name = "0065-14_1-409&525_508&580-499&580_409&557_418&525_508&548-0_0_20_30_21_24_24-78-32.jpg"
debug_dir = "data/ccpd"
font = ImageFont.truetype("./Font/platech.ttf", 20, 0)

# rect[x1,y1,x2,y2]
def draw_points(image,rect,plate):
    cv2.polylines(image,[np.array(rect)],True,(0,0, 255), 2,cv2.LINE_AA)
    x1 = rect[2][0]
    y1 = rect[2][1]
    cv2.rectangle(image,(x1-1,y1-20), (x1 + 200, y1), (0, 0, 255), -1,cv2.LINE_AA) # 字背景
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((x1+1, y1-20),plate, (255, 255, 255), font=font)
    imagex = np.array(img)
    return imagex

def parse_folder(folder,limit=None):
    files = os.listdir(folder)
    c = 0
    for f in files :
        name, ext = os.path.splitext(f)
        if ext.upper() != ".JPG": continue
        print("处理:",f)
        plate, points = parse(f)
        image = cv2.imread(os.path.join(folder,f))
        image = draw_points(image,points,plate)
        cv2.imwrite(os.path.join(debug_dir,f),image)
        c+=1
        if c>limit: break

def parse(name):

    data = name.split("-")
    desc = ["面积","倾斜","bbox","四点","车牌","明亮","模糊"]
    area = data[0]                                      # 0065
    tilt = data[1].split("_")                           # 14_1
    bbox =  [d.split("&") for d in data[2].split("_")]  # 409&525_208&580

    four_points = [d.split("&") for d in data[3].split("_")] # 499&580_409&557_418&525_508&548
    points = []
    for p in four_points:
        points.append([int(fp) for fp in p])

    plate =  data[4].split("_")                        # 0_0_20_30_21_24_24
    province = provinces[int(plate[0])]
    plate_id = "".join([ads[int(index)] for index in plate])
    plate = province + plate_id

    print("解析:",plate," ",points)

    return plate,points




if __name__ == '__main__':
    path = "/Volumes/pigindisk/CCPD2019/ccpd_weather"
    parse(f_name)
    parse_folder(path,10)