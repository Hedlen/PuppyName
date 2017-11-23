import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


#Inception-v3 model
BOTTLLENECK_TENSOR_SIZE=2048

BOTTLLENECK_TENSOR_NAME='pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME='DecodeJpeg/contents:0'
MODEL_DIR='./'

MODEL_FILE='tensorflow_inception_graph.pcde'
CACHE_DIR='./bottleneck'

INPUT_DATA='./train2'
INPUT_DATA_TEST='./test2'
Model_Path='./model.ckpt'
VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10

#neural network setting
LEARNING_RATE=0.01
STEPS=10000
BATCH=100

def creat_image_lists(testing_percentage,validation_percentage):
    result={}
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]

    is_root_dir=True

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
        extensions=['jpg','jpeg','JPG','JPEG']
        file_list=[]
        dir_name=os.path.basename(sub_dir)

        for extension in extensions:
            file_glob=os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))

        if not file_list:continue

        label_name=dir_name.lower()

        training_images=[]
        test_images=[]
        validating_images=[]

        for file_name in file_list:
            base_name=os.path.basename(file_name)
            chance=np.random.randint(100)
            if chance<validation_percentage:
                validating_images.append(base_name)
            elif chance<(testing_percentage+validation_percentage):
                test_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name]={
            'dir':dir_name,
            'training':training_images,
            'testing':test_images,
            'validation':validating_images,
        }
    return result
#def creat_image_test_lists(path_test_data):

def get_image_path(image_lists,image_dir,label_name,index,category):
    label_lists=image_lists[label_name]
    category_list=label_lists[category]

    mod_index=index % len(category_list)

    base_name=category_list[mod_index]
    sub_dir=label_lists['dir']
    full_path=os.path.join(image_dir,sub_dir,base_name)
    return full_path

def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'

def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    bottleneck_values=sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    bottleneck_values=np.squeeze(bottleneck_values)

    return bottleneck_values

def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    label_lists=image_lists[label_name]
    sub_dir=label_lists['dir']

    sub_dir_path=os.path.join(CACHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path):os.makedirs(sub_dir_path)
    bottleneck_path=get_bottleneck_path(image_lists,label_name,index,category)

    if not os.path.exists(bottleneck_path):
        image_path=get_image_path(
            image_lists,INPUT_DATA,label_name,index,category
        )
        image_data=gfile.FastGFile(image_path,'rb').read()

        bottleneck_values=run_bottleneck_on_image(
            sess,image_data,jpeg_data_tensor,bottleneck_tensor
        )

        bottleneck_string=','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string=bottleneck_file.read()
        bottleneck_values=[float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

def get_random_cached_bottlenecks(
    sess,n_classes,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    label_name_list=list(image_lists.keys())
    for _ in range(how_many):
        label_index=random.randrange(n_classes)
        label_name=list(image_lists.keys())[label_index]
        image_index=random.randrange(65536)

        bottleneck=get_or_create_bottleneck(
            sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor
        )
        ground_truth=np.zeros(n_classes,dtype=np.float32)
        ground_truth[label_index]=1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    label_name_list=list(image_lists.keys())

    for label_index,label_name in enumerate(label_name_list):
        category='testing'
        for index,unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck=get_or_create_bottleneck(
                sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor
            )
            ground_truth=np.zeros(n_classes,dtype=np.float32)
            ground_truth[label_index]=1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def main(_):
    accuracy=0
    image_lists=creat_image_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    label_name_list=list(image_lists.keys())
    with open('label.txt','w') as f_labels:
        for i in range(100):
            f_labels.write(label_name_list[i]+' ')
    n_classes=len(image_lists.keys())
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME]
    )
    bottleneck_input=tf.placeholder(
        tf.float32,[None,BOTTLLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder'
    )
    ground_truth_input=tf.placeholder(
        tf.float32,[None,n_classes],name='GroundTruthInput'
    )
    with tf.name_scope('final_training_ops'):
        weights=tf.Variable(tf.truncated_normal(
        [BOTTLLENECK_TENSOR_SIZE,n_classes],stddev=0.001
        ))
        biases=tf.Variable(tf.zeros([n_classes]))
        logits=tf.matmul(bottleneck_input,weights)+biases
        final_tensor=tf.nn.softmax(logits)

    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input,logits=logits)
    '''
    global_step=tf.Variable(0)
    LEARNING_RATE=tf.train.exponential_decay(
        0.1,global_step,100,0.96,staircase=True)
    '''
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    train_step=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    with tf.name_scope('evaluation'):
        correct_prediction=tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
        evaluation_step=tf.reduce_mean(
            tf.cast(correct_prediction,tf.float32)
        )
    saver=tf.train.Saver()
    with tf.Session() as sess:
        #init=tf.initialize_all_variables()
        init=tf.global_variables_initializer()
        sess.run(init)
        
        for i in range(STEPS):
            train_bottlenecks,train_ground_truth=get_random_cached_bottlenecks(
                sess,n_classes,image_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor
            )
            sess.run(train_step,feed_dict={bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truth})
            if i%100==0 or i+1==STEPS:
                validation_bottlenecks,validation_ground_truth=get_random_cached_bottlenecks(
                    sess,n_classes,image_lists,BATCH,'validation',jpeg_data_tensor,bottleneck_tensor
                )
                validation_accuracy=sess.run(
                    evaluation_step,feed_dict={
                        bottleneck_input:validation_bottlenecks,
                        ground_truth_input:validation_ground_truth
                    }
                )
                if accuracy<=validation_accuracy:
                    accuracy=validation_accuracy
                    saver.save(sess,Model_Path)
                print('Step %d:Validation accuracy on random sampled %d examples = %.lf%%' % (i,BATCH,validation_accuracy*100))
        saver.save(sess,'latest.ckpt')
        test_bottlenecks,test_ground_truth=get_test_bottlenecks(
            sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor
            )

        test_accuracy=sess.run(evaluation_step,feed_dict={
            bottleneck_input:test_bottlenecks,
            ground_truth_input:test_ground_truth
            })

        print('Final test accuray =%.lf%%' % (test_accuracy*100))
        
if __name__=='__main__':
    tf.app.run()
