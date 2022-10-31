import tensorflow 
import numpy 

MNIST_data=tensorflow.keras.datasets.mnist
(train_data,train_label),(test_data,test_label)=MNIST_data.load_data()
train_data=tensorflow.expand_dims(train_data/255.,-1) 
test_data=tensorflow.expand_dims(test_data/255.,-1)
train_label=numpy.float32(tensorflow.keras.utils.to_categorical(train_label,num_classes=10))
test_label=numpy.float32(tensorflow.keras.utils.to_categorical(test_label,num_classes=10))

Traindata=tensorflow.data.Dataset.from_tensor_slices((train_data,train_label)).batch(128).shuffle(128*10)
Testdata=tensorflow.data.Dataset.from_tensor_slices((test_data,test_label)).batch(128)

Data=tensorflow.keras.Input([28,28,1])
Conv1=tensorflow.keras.layers.Conv2D(32,3,padding='SAME',activation=tensorflow.nn.relu)(Data)
Conv2=tensorflow.keras.layers.Conv2D(64,3,padding='SAME',activation=tensorflow.nn.relu)(Conv1)
Pool=tensorflow.keras.layers.MaxPool2D(strides=[1,1])(Conv2)
Conv3=tensorflow.keras.layers.Conv2D(128,3,padding='SAME',activation=tensorflow.nn.relu)(Pool)
Conv4=tensorflow.keras.layers.Conv2D(128,3,padding='SAME',activation=tensorflow.nn.relu)(Conv3)
Flatten=tensorflow.keras.layers.Flatten()(Conv4)
Dense=tensorflow.keras.layers.Dense(256,activation=tensorflow.nn.relu)(Flatten)
Output=tensorflow.keras.layers.Dense(10,activation=tensorflow.nn.softmax)(Dense)
Minst_model=tensorflow.keras.Model(inputs=Data,outputs=Output)
print(Minst_model.summary())
#训练
Opt=tensorflow.optimizers.Adam(1e-3)
CE_Loss=tensorflow.losses.categorical_crossentropy
Minst_model.compile(optimizer=Opt,loss=CE_Loss,metrics=['accuracy'])
Minst_model.fit(Traindata,epochs=10)
#测试
Test_score=Minst_model.evaluate(Testdata)[1]
print('模型性能测试得分:',Test_score)
