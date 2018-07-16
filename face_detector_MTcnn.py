import tensorflow as tf
import numpy as np
import cv2
import detect_face



if __name__=="__main__":
    image_size=200
    minsize=20
    threshold=[0.6,0.7,0.7]
    factor = 0.709  # scale factor
    print("Creating MTcnn networks and load paramenters..")
    #########################build mtcnn########################
    with tf.Graph().as_default():
        sess=tf.Session()
        with sess.as_default():
            pnet,rnet,onet=detect_face.create_mtcnn(sess,'./model/')

    capture=cv2.VideoCapture(0)
    while(capture.isOpened()):
        ret,frame=capture.read()
        bounding_box,_=detect_face.detect_face(frame,minsize,pnet,rnet,onet,threshold,factor)

        nb_faces=bounding_box.shape[0]#人脸检测的个数
        #标记人脸
        for face_position in bounding_box:
            rect=face_position.astype(int)
            #矩形框
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,255),2,1)
            cv2.putText(frame,"faces:%d"%(nb_faces),(10,20),cv2.FONT_HERSHEY_COMPLEX,1,(255, 0, 255), 4)



        cv2.imshow('Video',frame)
        if cv2.waitKey(1)& 0xff==27:
            break
    capture.release()
    cv2.destroyAllWindows()




