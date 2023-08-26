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
dev = 0.0


model_name="ARAE_QM9"
save_dir="./save/"+model_name
Ntest=10000
num_test_batches = int(Ntest/batch_size)


model = ARAE(vocab_size = vocab_size,
             batch_size = batch_size,
             latent_size = latent_size,
             sample_size = sample_size,
             )


out_dir0="out_"+model_name
if not os.path.exists(out_dir0):
    os.makedirs(out_dir0)

total_st=time.time()

epochs=[79]

for epoch in epochs:
    out_dir=out_dir0+"/%d" %epoch
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file=out_dir+"/result_"+model_name+"_%d.txt" %epoch
    fp0=open(output_file,"w")
    model.restore(save_dir+"/model.ckpt-%d" %epoch)

    latent_vector_fake=[]
    Y_fake=[]
    smiles_fake=[]

    for itest in range(num_test_batches):


        decoder_state = model.get_decoder_state()
        s = np.random.normal(0.0, 0.25, [batch_size, sample_size]).clip(-1.0,1.0)
        latent_vector = model.generate_latent_vector(s)
        latent_vector_fake.append(latent_vector)

        start_token = np.array([char_list.index('X') for i in range(batch_size)])
        start_token = np.reshape(start_token, [batch_size, 1])
        length = np.array([1 for i in range(batch_size)])
        smiles = ['' for i in range(batch_size)]
        Y=[]
        for i in range(seq_length):
            m, state = model.generate_molecule(start_token, latent_vector, length, decoder_state)
            decoder_state = state
            start_token = np.argmax(m,2)
            Y.append(start_token[:,0])
            smiles = [s + str(char_list[start_token[j][0]]) for j,s in enumerate(smiles)]
        Y=list(map(list,zip(*Y)))
        Y_fake.append(Y)
        smiles_fake+=smiles


    latent_vector_fake=np.array(latent_vector_fake,dtype="float32").reshape(-1,latent_size)
    Y_fake=np.array(Y_fake,dtype="int32").reshape(-1,seq_length)
    outfile=out_dir+"/Zfake.npy"
    np.save(outfile,latent_vector_fake)
    outfile=out_dir+"/Yfake.npy"
    np.save(outfile,Y_fake)

    outfile=out_dir+"/smiles_fake.txt"
    fp_out=open(outfile,'w')
    for line in smiles_fake:
        line_out=line+"\n"
        fp_out.write(line_out)
    fp_out.close()

total_et=time.time()
print ("total_time : ", total_et-total_st)

