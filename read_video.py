import cv2
import numpy as np

cap = cv2.VideoCapture('gorev1_video.mp4') 

object_detector = cv2.createBackgroundSubtractorMOG2()

rect_color = (0, 0,255) #BGR olarak kırmızı

#gorev1_video.mp4 480*640 olması ve istenen yüzdelere göre ayarlanmıştır.
target_x = 160
target_y = 48
target_width = 320
target_height = 384

while (True):
    ret, frame = cap.read()
    height, weight, _ = frame.shape

    #Görevde istenen bölgeyi tayin ediyoruz.
    #Hedef Vuruş Alanında iken kırmızı kutuya al Kilitlenme Dörtgeni yazdır

    mask = object_detector.apply(frame)

    #Hareket eden daireyi tespit edelim
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if target_x < x < target_x + target_width and target_y < y < target_y +target_height:
            print("Frame'i" ,x/weight*100, "yatayda, yüzde" ,y/height*100, "dikeyde kapsayan hedef, Hedef Vuruş Alani'nda.")

            #Hedef Kilitlenme Diktörtgenini çizdir
            cv2.rectangle(frame, (target_x, target_y), (target_x + target_width, target_y + target_height), rect_color, 5)
            cv2.putText(frame, "Av : Hedef Vurus Alani", (target_x +10, target_y + target_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

            # Kilitlenme Dörtgeni çizdir
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, "Kilitlenme Dortgeni", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)
            

    cv2.imshow('output', frame)

    if(cv2.waitKey(10) & 0xFF == ord('a')) :
        break

cap.release()
cv2.destroyAllWindows()
