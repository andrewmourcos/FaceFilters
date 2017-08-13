import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
helmet = cv2.imread('helmet.png')


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        h = h+150
        w= w+150
        roi_color = img[y-100:y -100+ h, x:x + w]

        resize_helmet = cv2.resize(helmet, (w,h), interpolation=cv2.INTER_AREA)

        img2gray = cv2.cvtColor(resize_helmet, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 2, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi_color, roi_color, mask=mask_inv)
        img2_fg = cv2.bitwise_and(resize_helmet, resize_helmet, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)
        img[y-100:y -100+ h, x:x + w] = dst

        cv2.imshow('image',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
