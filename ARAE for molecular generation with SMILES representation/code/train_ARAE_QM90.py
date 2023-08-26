from model.ARAE import ARAE
#from utils.utils import *
import numpy as np
import os, sys
import time
import tensorflow as tf
import collections
from six.moves import cPickle
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

def convert_to_smiles(vector, char):
    smiles=""
    for i in vector:
        smiles+=char[i]
    return smiles

def cal_accuracy(S1, S2, length):
    count = 0
    for i in range(len(S1)):
        if np.array_equal(S1[i][1:length[i]+1],S2[i][:length[i]]):
            count+=1
    return count

def read_data(data_dir,data):

    Xfile=data_dir+"X"+data+".npy"
    Yfile=data_dir+"Y"+data+".npy"
    Lfile=data_dir+"L"+data+".npy"

    Xdata=np.load(Xfile)
    Ydata=np.load(Yfile)
    Ldata=np.load(Lfile)

    return Xdata,Ydata,Ldata

char_list= ["H","C","N","O","F",
"n","c","o",
"1","2","3","4","5",
"(",")","[","]",
"-","=","#","+","X","Y"]

char_dict={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 
'n': 5, 'c': 6, 'o': 7, 
'1': 8, '2': 9, '3': 10, '4': 11, '5': 12, 
'(': 13, ')': 14, '[': 15, ']': 16, 
'-': 17, '=': 18, '#': 19, '+': 20, 'X': 21, 'Y': 22}

vocab_size = len(char_list)
latent_size = 200
batch_size = 100
sample_size = 100
seq_length = 34
dev1=0.2
dev0= 0.02
dev = dev0+dev1
dev_decay_rate = 0.995
num_epochs    = 80
learning_rate = 0.00003
temperature = 1.0
min_temperature = 0.5
decay_rate    = 0.95
coord = tf.train.Coordinator()

data_dir="./data/QM9"
save_dir="./save/ARAE_QM9"
model_name="ARAE_QM9"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Training     %s" %model_name )
print("vocab_size : %d" %vocab_size )
print("latent_size: %d" %latent_size)
print("batch_size : %d" %batch_size )
print("sample_size: %d" %sample_size)
print("seq_length : %d" %seq_length )
print("num_epochs : %d" %num_epochs )
print("dev        : %f" %dev        )
print("decay_rate : %f" %decay_rate )
print("dev_decay  : %f" %dev_decay_rate)


Xtrain,Ytrain,Ltrain=read_data(data_dir,"train")
Xtest,Ytest,Ltest=read_data(data_dir,"test")

model = ARAE(vocab_size = vocab_size,
             batch_size = batch_size,
             latent_size = latent_size,
             sample_size = sample_size,
             )


num_train_batches = int(len(Xtrain)/batch_size)
#num_train_batches = 10
index=np.arange(0,len(Xtrain))
np.random.shuffle(index)
Ntrain=num_train_batches*batch_size
index=index[0:Ntrain]

Xtrain = Xtrain[index]
Ytrain = Ytrain[index]
Ltrain = Ltrain[index]
xbatches = np.split(Xtrain, num_train_batches, 0)
ybatches = np.split(Ytrain, num_train_batches, 0)
lbatches = np.split(Ltrain, num_train_batches, 0)

num_test_batches = int(len(Xtest)/batch_size)

Xtest = Xtest[0:num_test_batches*batch_size]
Ytest = Ytest[0:num_test_batches*batch_size]
Ltest = Ltest[0:num_test_batches*batch_size]
xtest_batches = np.split(Xtest, num_test_batches, 0)
ytest_batches = np.split(Ytest, num_test_batches, 0)
ltest_batches = np.split(Ltest, num_test_batches, 0)

total_st=time.time()

#model.restore(save_dir+"/model.ckpt-39")

n_iter_gan=1
for epoch in range(0,num_epochs):
    if epoch==2 or epoch==4 or epoch==6:
        n_iter_gan+=1
    # Learning rate scheduling 
    model.assign_lr(learning_rate * (decay_rate ** epoch))
    st=time.time()
    print("epoch : ", epoch)

    itest=-1
    for iteration in range(num_train_batches):
        total_iter = epoch*num_train_batches + iteration

        x = xbatches[iteration]
        y = ybatches[iteration]
        l = lbatches[iteration]

#        s = np.random.normal(0.0, 1.0, [batch_size, sample_size])
        model.train(x,y,l,n_iter_gan,dev)

        if iteration % 200 == 0:
            itest+=1
            x = xtest_batches[itest]
            y = ytest_batches[itest]
            l = ltest_batches[itest]

#            s = np.random.normal(0.0, 1.0, [batch_size, sample_size])
            s = np.random.normal(0.0, 0.25, [batch_size, sample_size]).clip(-1.0,1.0)
#            n = np.random.normal(0.0, dev, [batch_size, latent_size])
            n = np.random.normal(0.0, 0.0, [batch_size, latent_size])

            Y, cost1, cost2, cost3, cost4, latent_vector, mol_encoded0 = model.test(x, y, l, s, n)
            norm = np.linalg.norm(latent_vector, 2, axis=-1)
            print ('norm check : ', norm[0:5])
            line_out="test_iter : %d, epoch %d, cost: %f %f %f %f" %(iteration,epoch,cost1,cost2,cost3,cost4)
            print(line_out)

            accuracy_ = cal_accuracy(x, Y, l)
            line_out="accuracy: %f " %(accuracy_)
            print(line_out)

            for i in range (1):
                s1_1 = convert_to_smiles(x[i,:], np.array(char_list))
                s1_2 = convert_to_smiles(Y[i,:], np.array(char_list))
                print (s1_1[1:]+"\n"+s1_2[:-1])

                #print (test_length[i], len(s1_1.replace(' ', '')))
                #print (s1_1+"\n"+s1_2[:test_length[i]])

        if iteration %400 == 0:
            decoder_state = model.get_decoder_state()
            print ('**********************************************')

#            s = np.random.normal(0.0, 1.0, [batch_size, sample_size])
            s = np.random.normal(0.0, 0.25, [batch_size, sample_size]).clip(-1.0,1.0)


            latent_vector = model.generate_latent_vector(s)
            start_token = np.array([char_list.index('X') for i in range(batch_size)])
            start_token = np.reshape(start_token, [batch_size, 1])
            length = np.array([1 for i in range(batch_size)])
            smiles = ['' for i in range(batch_size)]
            for i in range(seq_length):
                m, state = model.generate_molecule(start_token, latent_vector, length, decoder_state)
                decoder_state = state
                start_token = np.argmax(m,2)
                smiles = [s + str(char_list[start_token[j][0]]) for j,s in enumerate(smiles)]
            for s in smiles[:10]:
                print (s)

        if iteration%300==0 and iteration!=0:
            dev1*=dev_decay_rate
            dev = dev0+dev1
            print ('deviation : ', dev)

    ckpt_path = save_dir+'/model.ckpt'
    model.save(ckpt_path, epoch)
    # Save network!
    et = time.time()
    print ("time : ", et-st)

total_et=time.time()

print ("total_time : ", total_et-total_st)

#ckpt_path = save_dir+'/model.ckpt'
#model.save(ckpt_path,total_iter)


