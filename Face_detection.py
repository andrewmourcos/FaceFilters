# include library for computer vision
import cv2

# include Haar Cascade (trained sample data) to compare to faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# include "filter"
helmet = cv2.imread('helmet.png')

# get video input
cap = cv2.VideoCapture(0)

while 1:
    # get frames
    ret, img = cap.read()
    # convert frames to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'par Andrew Mourcos', (0, 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # iterate over faces found
    for (x, y, w, h) in faces:
        h = h+150
        w = w+150
        roi_color = img[y-100:y -100 + h, x:x + w]
        resize_helmet = cv2.resize(helmet, (w,h), interpolation=cv2.INTER_AREA)

        try:
            img2gray = cv2.cvtColor(resize_helmet, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 2, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img1_bg = cv2.bitwise_and(roi_color, roi_color, mask=mask_inv)
            img2_fg = cv2.bitwise_and(resize_helmet, resize_helmet, mask=mask)
            dst = cv2.add(img1_bg, img2_fg)
            img[y-100:y -100 + h, x:x + w] = dst
        except:
            pass

        cv2.imshow('image',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
