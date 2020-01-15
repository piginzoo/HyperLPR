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
import sys,os
sys.path.insert(0, "../Mobilenet-SSD-License-Plate-Detection")
import detect_opencv as do

fontC = ImageFont.truetype("./Font/platech.ttf", 50, 0)
dir = "/Users/piginzoo/Downloads/train_images/车牌/车贷业务给的照片"
label = "/Users/piginzoo/Downloads/train_images/车牌/车贷业务给的照片/label.txt"
debug_dir = "data/debug"


def SpeedTest(image_path):
    grr = cv2.imread(image_path)
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    model.SimpleRecognizePlateByE2E(grr)
    t0 = time.time()
    for x in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0)/20.0
    print("Image size :" + str(grr.shape[1])+"x"+str(grr.shape[0]) +  " need " + str(round(t*1000,2))+"ms")

# rect[x1,y1,x2,y2]
def drawRectBox(image,rect,addText):
    x1,y1,x2,y2 = rect
    cv2.rectangle(image,(x1,y1),(x2,y2), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image,(x1-1,y1-50), (x1 + 600, y1), (0, 0, 255), -1,cv2.LINE_AA) # 字背景
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((x1+1, y1-50),addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex

def detect_ssd(image,pr):

    bboxes,small_images,scores = do.detect(image)

    plate_texts = []
    confidences = []
    for img in small_images:
        plate_text,confidence = pr.recognizeOne(img)
        plate_texts.append(plate_text)
        confidences.append(confidence)

    return bboxes,small_images,plate_texts,scores,confidences



def recongniz(f,label):
    f_full_path = os.path.join(dir, f)
    grr = cv2.imread(f_full_path)

    print("处理图片：", f)

    bboxes, small_images, plate_texts, scores,confidences = detect_ssd(grr,model)


    print("识别出%d个车牌" % len(bboxes))
    correct=False
    for i,bbox in enumerate(bboxes):
        small_image = small_images[i]
        plate_text = plate_texts[i]
        confidence = confidences[i]
        score = scores[i]
        rect = [int(r) for r in bbox]

        cv2.imwrite(os.path.join(debug_dir, f+"_"+label + "_" + \
                                 str(round(score, 3)) + "_" + \
                                 str(round(confidence, 3)) + \
                                 ".jpg"), small_image)
        grr = drawRectBox(grr, rect, plate_text + " " + str(round(score, 3)) + "/" + str(round(confidence, 3)))
        print("----------------------")
        print("标签:",label)
        print("车牌:",plate_text)
        print("检测置信:", score)
        print("文字置信:", confidence)

        if label==plate_text: correct=True

    cv2.imwrite(os.path.join(debug_dir, f), grr)
    print("=======================================================================================")
    return correct

if __name__ == '__main__':


    print(sys.argv)
    file_name = None
    if len(sys.argv)==2:
        file_name = sys.argv[1]
    if len(sys.argv)==3:
        file_name = sys.argv[1]
        label = sys.argv[2]

    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")

    if file_name is not None:
        print("识别：",file_name,":",label)
        recongniz(file_name,label)
    else:
        print("识别所有的图片")
        correct=0
        with open(label,"r") as f:
            lines = f.readlines()
            for line in  lines:
                f,label = line.split()
                name,ext = os.path.splitext(f)
                if ext.upper()!=".JPG": continue

                if recongniz(f,label):correct+=1
            print("识别正确率：",(correct/len(lines)))