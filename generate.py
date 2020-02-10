import cv2
import sys

img = cv2.imread(sys.argv[1])
i=0
for x in range(0, 755, 7):
    for y in range(0, 335, 7):
        crop_img = img[y:y+84, x:x+84]
        filename=str(i)+'.jpg'
        i=i+1
        cv2.imwrite(filename, crop_img)
