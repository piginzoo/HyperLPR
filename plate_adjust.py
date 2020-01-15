import cv2
image = cv2.imread("data/plate.jpg")
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
image = cv2.GaussianBlur(image,(3,3),0)
x = cv2.Sobel(image,cv2.CV_16S,1,0)  
y = cv2.Sobel(image,cv2.CV_16S,0,1)  
absX = cv2.convertScaleAbs(x)   # 转回uint8  
absY = cv2.convertScaleAbs(y)  
image = cv2.addWeighted(absX,0.5,absY,0.5,0)  
cv2.imwrite("data/debug/sobel.jpg",image)
ret,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("data/debug/binary.jpg",image)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=5)    #闭运算1
# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=3)     #开运算1
cv2.imwrite("data/debug/done.jpg",image)
