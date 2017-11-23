# -*- coding: utf-8 -*-    
import os.path
import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
import glob
#Inception-v3 model
BOTTLLENECK_TENSOR_SIZE=2048

BOTTLLENECK_TENSOR_NAME='pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME='DecodeJpeg/contents:0'
MODEL_DIR='./'

MODEL_FILE='tensorflow_inception_graph.pcde'
Model_Path='model.ckpt'
#MODEL_DIR='./'
#MODEL_FILE='model.ckpt.meta'
VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10
#TEST_DATA_PATH='./test2/test'
TEST_DATA_PATH='./test2/test'
TRAIN_DATA_PATH='./train2'
#neural network setting
LEARNING_RATE=0.01
STEPS=4000
BATCH=100
def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    bottleneck_values=sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    bottleneck_values=np.squeeze(bottleneck_values)

    return bottleneck_values
def get_image_lists():
    result=[]
    sub_dirs=[x[0] for x in os.walk(TEST_DATA_PATH)]
    for sub_dir in sub_dirs:
        extensions=['jpg','jpeg','JPG','JPEG']
        file_list=[]
        for extension in extensions:
            file_glob=os.path.join(TEST_DATA_PATH,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:continue
        test_images=[]
        for file_name in file_list:
            base_name=os.path.basename(file_name)
            test_images.append(base_name)
        result=test_images
    return result
def get_class_labels():
    label_list=[]
    with open('label.txt','r') as f_read:
        label=f_read.readlines()
        for label_num in label:
            words=label_num.split()
            for i in range(100):
                label_list.append(words[i])
    return label_list
def get_or_create_bottleneck_new(sess, image_lists,image_dir,index,jpeg_data_tensor, bottleneck_tensor):
    
    mod_index=index % len(image_lists)
    base_name=image_lists[mod_index]
    image_path=os.path.join(image_dir,base_name)
    image_data=gfile.FastGFile(image_path,'rb').read()
    bottleneck_values=run_bottleneck_on_image(
        sess,image_data,jpeg_data_tensor,bottleneck_tensor
    )
    return bottleneck_values
def image_Classfier(bottleneck_tensor, jpeg_data_tensor):
    test_file=r'./'
    if os.path.isfile('./test_final.txt'):
        os.remove('./test_final.txt')
    f=open(test_file+r"label2.txt",'a')
    #f.write('Imagefile_ID'+'  '+'class_ID')
    #f.close()
    image_lists = get_image_lists()
    n_classes = 100

    class_label=get_class_labels()

    bottleneck_input=tf.placeholder(
        tf.float32,[None,BOTTLLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder'
    )
    '''
    ground_truth_input=tf.placeholder(
        tf.float32,[None,n_classes],name='GroundTruthInput'
    )
    '''
    with tf.name_scope('final_training_ops'):
        weights=tf.Variable(tf.truncated_normal(
        [BOTTLLENECK_TENSOR_SIZE,n_classes],stddev=0.001
        ))
        biases=tf.Variable(tf.zeros([n_classes]))
        logits=tf.matmul(bottleneck_input,weights)+biases
        #final_tensor=tf.nn.softmax(logits)
        image_order_step = tf.argmax(logits, 1)
    '''
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
       '''
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        #if os.path.exists(Model_Path):
        saver.restore(sess, Model_Path)
            #print("ckpt file already exist!")
        #with tf.name_scope('kind'):
        # image_kind=image_lists.keys()[tf.arg_max(final_tensor,1)]
            #image_order_step = tf.arg_max(final_tensor, 1)
        #label_name_list = list(image_lists.keys())
        #for label_index, label_name in enumerate(label_name_list):
            #category = 'testing'
        #with open("%s.txt"%title,"a") as f:
        for index, unused_base_name in enumerate(image_lists):
            #for index, unused_base_name in enumerate(image_lists[category]):
            bottlenecks = []
                #ground_truths = []
                #print("真实值%s:" % label_name)
                # print(unused_base_name)
            bottleneck = get_or_create_bottleneck_new(sess, image_lists,TEST_DATA_PATH,index,jpeg_data_tensor, bottleneck_tensor)
                #ground_truth = np.zeros(n_classes, dtype=np.float32)
                #ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
                #ground_truths.append(ground_truth)
            image_kind = sess.run(image_order_step, feed_dict={
                bottleneck_input: bottlenecks})
            image_kind_order = int(image_kind[0])
            unused_base_name=unused_base_name.strip('.jpg')
            f.write(class_label[image_kind_order]+'\t'+unused_base_name+'\n')
            print("Image ID：%s,Predictive value:%s:"% (unused_base_name,class_label[image_kind_order]))
        print("Test result have been written to the label.txt file，please go to the specified path to view!")
        f.close()
def main():
     with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(
            graph_def,
            return_elements=[BOTTLLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME]
            )
     #global TEST_DATA_PATH
     #print("请输入需要测试的图片文件的根路径！！！")
     #TEST_DATA_PATH=raw_input("Please input test image file path:")
     #if not os.path.exists(TEST_DATA_PATH):
     #    print("输入的路径不存在！！")
     #    print("请输入如/home/hedlen/python/project/test的路径方式")
     #    sys.exit(0)
     image_Classfier(bottleneck_tensor, jpeg_data_tensor)

if __name__ == '__main__':
    main()