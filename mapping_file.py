import os
import shutil
import math
import h5py
num_class=134
path_to_train_dataset_dir='train'
path_to_val_dataset_dir='val'
path_map_dir='train3'
def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
def file_process(path_file_dir): 
    train_filenames = os.listdir(path_file_dir)
    path_file_length_num= len(train_filenames)
    with open('data_'+path_file_dir+'_image.txt','r')  as f :
        f_train_set_jpg= open(path_file_dir+"_set_jpg_id.txt","w")
        for i in range(path_file_length_num):
            train_set_jpg_id= train_filenames[i].rstrip('.jpg')
            f_train_set_jpg.write(train_set_jpg_id+'\n')
        
        data_train_image_file = f.readlines()   
        length_train_image = len(data_train_image_file)
        f_train_set= open(path_file_dir+"_set_data.txt","w") 
        f_train_labels= open(path_file_dir+"_labels.txt","w") 
        for line in data_train_image_file:
            words = line.split()
            f_train_set.write( str(words[0]))
            f_train_set.write('\n')
            f_train_labels.write( str(words[1]))
            f_train_labels.write('\n')
        f_train_set.close()   
        f_train_labels.close() 
        f_train_set_jpg.close()
    if path_file_dir=='train':
        rmrf_mkdir(path_map_dir)
        for i in range(num_class):
            os.mkdir(os.path.join(path_map_dir,str(i)))
    with open(path_file_dir+'_set_data.txt','r')  as f0:
        train_set_data= f0.readlines()
        train_set_data_num= len(train_set_data)
    with open(path_file_dir+'_labels.txt','r')  as f1:
        train_labels= f1.readlines()
        train_labels_num= len(train_labels)
    with open(path_file_dir+'_set_jpg_id.txt','r')  as f2:
        train_set_jpg_id= f2.readlines()
        train_set_jpg_id_num= len(train_set_jpg_id)
    for index_formal in range(train_set_data_num):
        for index_fault in range(train_set_jpg_id_num):
            if train_set_jpg_id[index_fault] == train_set_data[index_formal]:
                label_number =  train_labels[index_formal]
                label_number=label_number.rstrip('\n')
                filename=train_set_jpg_id[index_fault].rstrip('\n')+'.jpg'
                if path_file_dir=='train':
                    os.symlink('../../train/'+filename,path_map_dir+'/'+str(int(label_number))+'/'+filename)
                else:
                     os.symlink('../../val/'+filename,path_map_dir+'/'+str(int(label_number))+'/'+filename)
    
rmrf_mkdir('test2')
os.symlink('../image/', 'test2/test')
print('Done!')
if __name__=='__main__':
    file_process(path_to_train_dataset_dir)
    file_process(path_to_val_dataset_dir)