# PyTorch_Medical_BiLSTM_CRF_NER

### datas文件夹中：**

1. The storage in **datas** is the original data, including two files, as follows:
   - **Ruijin_round1_train_smal**l  stores our training data, and each .txt file corresponds to a .ann file.
   - **.txt file** is a document containing a lot of words.
   - The .**ann file** contains the entity class contained in the **.txt file** and the start and end locations of the entity
2. **Ruijin_round1_test_a_20181022** I did not use the data in this file because I only trained with small data (the model is not perfect) and may use it later, so save it.

### data文件夹中：

This folder contains dictionaries, training data, and validation data, stored as .pkl.



### Introduction:

I don’t know if it ’s the cause of memory. When I use <u>**pickle.dump ()**</u> to store data, I keep getting errors, but when I use very small data, it can run normally. This is why I use small data. .
Data segmentation and data cleaning are written in great detail. Only one long document in this one text is very applicative, I hope it can be referenced.
Because the next time will be fully allocated to other work and time constraints, I simply wrote the BiLSTM model (this model is also very sloppy, because it only provides readers with ideas), if there is time later, then fully Write this aspect; the training is also written briefly (after all, the small data is not enough to explain any problems, only for testing)





