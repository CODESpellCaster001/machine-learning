import time
import numpy as np
np.random.seed(10) #可复现
import tensorflow as tf

cifar10 = tf.keras.datasets.cifar10#提取
(x_img_train,y_label_train), (x_img_test, y_label_test)=cifar10.load_data()
print("train data:",'images:',x_img_train.shape,
" labels:",y_label_train.shape)
print("test  data:",'images:',x_img_test.shape ,
      " labels:",y_label_test.shape)
label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
#显示前25个images的label
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_img_train[i], cmap=plt.cm.binary)
    plt.xlabel(label_dict[y_label_train[i][0]])
plt.show()
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
y_label_train_OneHot = tf.keras.utils.to_categorical(y_label_train)
y_label_test_OneHot = tf.keras.utils.to_categorical(y_label_test)


y_label_test_OneHot.shape
# Model definition
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                 input_shape=(32, 32, 3),
                 activation='relu',
                 padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.25))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.40))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(1024, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(rate=0.30))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#查看卷积神经网络摘要
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

t1=time.time()
train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.2,
                        epochs=30, batch_size=128, verbose=1)
t2=time.time()
CNNfit = float(t2-t1)
print("Time taken: {} seconds".format(CNNfit))


scores = model.evaluate(x_img_test_normalize,
                        y_label_test_OneHot, verbose=0)

scores[1]
