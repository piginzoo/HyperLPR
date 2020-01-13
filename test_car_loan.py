'''
    测试代码，for python3.x
'''
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import time
import HyperLPRLite as pr
import cv2
import numpy as np

fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

def SpeedTest(image_path):
    grr = cv2.imread(image_path)
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    model.SimpleRecognizePlateByE2E(grr)
    t0 = time.time()
    for x in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0)/20.0
    print("Image size :" + str(grr.shape[1])+"x"+str(grr.shape[0]) +  " need " + str(round(t*1000,2))+"ms")


def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex

if __name__ == '__main__':
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")

    dir = "/Users/piginzoo/Downloads/train_images/车牌/车贷业务给的照片"
    label = "/Users/piginzoo/Downloads/train_images/车牌/车贷业务给的照片/label.txt"
    debug_dir = "data/debug"
    import os

    with open(label,"r") as f:
        lines = f.readlines()
        for line in  lines:
            f,label = line.split()
            name,ext = os.path.splitext(f)
            if ext.upper()!=".JPG": continue

            f_full_path = os.path.join(dir, f)
            grr = cv2.imread(f_full_path)

            print("处理图片：",line)
            results = model.SimpleRecognizePlateByE2E(grr)
            print("识别出%d个车牌" % len(results))
            for pstr, confidence, rect in results:
                rect = [int(r) for r in rect]
                #x1,y1,x2,y2
                rect[1]-= 20
                rect[3]+= 20
                small_image = model.cropImage(grr, rect)

                cv2.imwrite(os.path.join(debug_dir, label + "_" + str(round(confidence, 3)) + ".jpg"), small_image)

                if confidence > 0.7:
                    image = drawRectBox(grr, rect, pstr + " " + str(round(confidence, 3)))
                    print("标签:",label)
                    print("车牌:",pstr)
                    print("置信:",confidence)
                else:
                    print("识别置信度太低了：标签=",label,",文件=",f)
                print("----------------------")
            print("==============================================")