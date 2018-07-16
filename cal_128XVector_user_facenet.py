# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import facenet
import pandas as pd

def build_facenet_model(modir='./model/20170512-110547.pb'):
    tf.Graph().as_default()
    sess=tf.Session()
    facenet.load_model(modir)

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    print('facenet embedding模型建立完毕')
    return sess, images_placeholder, phase_train_placeholder, embeddings

#通过facenet模型 计算图片的128向量
def cal_128_vector(frame,sess, images_placeholder, phase_train_placeholder, embeddings):
    scaled_reshape=[]
    embeddings_size=embeddings.get_shape()[1]
    #frame=cv2.imread(frame)#如果fram是路径的话 去掉注释
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image=cv2.resize(frame,(200,200))
    image=facenet.prewhiten(image)
    scaled_reshape.append(image.reshape(-1,200,200,3))
    array=np.zeros((1,embeddings_size))
    array[0,:]=sess.run(embeddings,feed_dict={images_placeholder: scaled_reshape[0],
                                                    phase_train_placeholder: False })[0]

    return array

def cal_dist(array0,array1):
    dist = np.sqrt(np.sum(np.square(array0[0] - array1[0])))
    return dist

#将128向量保存到csv文件中去
def saver_data_to_csv(array,label='lijie2',csv_dir='./data/data.csv'):
    # data1=DataFrame(array,index=None,columns=[label])
    # data1.to_csv(csv_dir)
    array=array[0,:]
    info=pd.read_csv(csv_dir)
    #print(info.shape)
    info[label]=array
    #print(info.shape)
    info.to_csv(csv_dir,index=None)
    return info

#计算两个数组之间的距离  并返回距离最小值  和 对应的标签
def cal_dist_from_csv(csv_dir,array):
    array1=array[0,:]
    final_column='others'#如果不满足最小距离  返回 others
    pre_dist=1
    info=pd.read_csv(csv_dir)
    #print(info.head(0))
    for i,column in enumerate(info.head(0)):
        array2=info[column]
        dist=cal_dist(array1,array2)
        if dist<pre_dist and dist<=0.5: #判断最相似的人脸
            pre_dist=dist
            final_column=column
    #print("final_column:",final_column)
    #print("final_dist:", pre_dist)
    return  pre_dist,final_column












if __name__=="__main__":

    image_name0 = './picture/1.jpg'  # change to your image name
    image_name1 = './picture/0.jpg'  # change to your image name

    #调用facenet模型
    sess, images_placeholder, phase_train_placeholder, embeddings=build_facenet_model()
    # 计算128 向量
    array0 = cal_128_vector(image_name0, sess, images_placeholder, phase_train_placeholder, embeddings)
   # print("the resulrt :", array0)

    #计算128 向量
    array1=cal_128_vector(image_name1,sess, images_placeholder, phase_train_placeholder, embeddings)
    #print("the resulrt :",array1)
    dist=cal_dist(array0,array1)
    print("dist:",dist)
    print("the array0 shape:",array0.shape)

    csv_dir = './data/data.csv'
    #saver_data_to_csv(array0,label='wangzheng',csv_dir)  #保存图片的128向量数据到 csv文件中去
    dist,column=cal_dist_from_csv(csv_dir,array1)       #计算array1 与csv文件中距离最近的距离 并返回对应标签
    print("dist is :%.2f"%(dist))
    print("the name is :",column)





