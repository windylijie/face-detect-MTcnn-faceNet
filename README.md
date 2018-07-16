# face-detect-MTcnn-faceNet
工作平台：
windos
tensorflow
pycharm
opencv

第一步：实现人脸检测功能（MTcnn）
最开始的时候使用的是opencv自带的人脸级联分类器进行人脸检测，但是 后来查一下资料说MTcnn能够实现的效果更好 ，所以尝试着用一下。
  
  对于MTcnn 框架 是存在 PNet RNet ONet 三个网络架构级联成的，由于电脑渣渣 所以直接下载别人训练好的库  存在mode/文件夹下 

（1）detect_face.py 文件实现了MTcnn人脸检测的相关函数



（2）face_detector_MTcnn.py 是对detect_face.py进行测试：实现了视频流下人脸的检测和定位功能


第二部：利用faceNet 实现两张图片距离向量的计算

（1）facenet.py 是我直接在网上下载的 文件 他实现了相关函数的处理具体faceNet的实现原理这里就不讲了 可以自行收索资料了解一下 我当时是看了吴恩达的视屏才知道这个方法， 开始的时候我只使用直方图比较 效果很差 所以这个faceNet方法很好

（2）cal_128XVector_user_facenet.py 文件是根据facenet.py文件里相关函数，计算出两张图片的distance：


      1、build_facenet_model（modir='./model/20170512-110547.pb'）函数： 是建立faceNet模型用的   由于电脑渣渣加上图片数据难找， 没法实现模型的训练我就下载了一个文件，（20170512-110547.pb） 可以在官网下载


      2、cal_128_vector（）函数： 就是计算一张图片的distance向量  计算结果会产生[1,128]数组 


      3、cal_dist（）函数： 计算两个数组的方差和 ，是根据这个结果来衡量两张图片的相似度
      
      
      4、saver_data_to_csv(array,label='lijie2',csv_dir='./data/data.csv'):将采集的图片放到 csv 文件中 label 作为该数据的标签 后面可以通过标签来判别识别的人脸的属性
      
      
      5、cal_dist_from_csv（）：计算 实时采集的图片 与 已经存在csv的数据对比 ，输出结果是 该实时图片与CSV文件中相似的 图片的distance 和标签



第三步：collect_frame_to_csv.py实时检测人脸，采集人脸数据到csv文件中去  也就是存储用户的人脸信息（faceNet + opencv ）

      具体实现方法 在程序中注释的很详细  可以直接运行使用
      
      
      
第四部：









第一个文件 collect_frame_to_csv.py:
主要实现的功能：利用MTcnn进行人脸检测，将人脸通过facenet 模型计算其128维向量，存在/data/data.csv文件中去,成为后续对比的数据



