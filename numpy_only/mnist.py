import gzip
import struct
import random
import numpy as np

with gzip.open('train-labels-idx1-ubyte.gz','rb') as f:
    f_label = f.read()
label = []
for i in f_label:
    label.append(i)
label = label[8:]

#print(label)
with gzip.open('train-images-idx3-ubyte.gz','rb') as f:
    f_image = f.read()
image = []
for i in range(16,len(f_image),784):
    temp = []
    for j in range(784):
        temp.append(f_image[i+j])
    temp = np.array(temp)
    thisLabel = np.zeros((10))
    thisLabel[ int( label[int((i-16)/784)] ) ] = 1
    temp = [temp, thisLabel]
    image.append(temp)

with gzip.open('t10k-labels-idx1-ubyte.gz','rb') as f:
    f_tlabel = f.read()
tlabel = []
for i in f_tlabel:
    tlabel.append(i)
tlabel = tlabel[8:]

with gzip.open('t10k-images-idx3-ubyte.gz','rb') as f:
    f_timage = f.read()
timage = []
for i in range(16,len(f_timage),784):
    temp = []
    for j in range(784):
        temp.append(f_timage[i+j])
    timage.append(temp)

print("Read Finished")

import autodiff as ad

x = ad.Variable(name = "x")
W = ad.Variable(name = "W")
b = ad.Variable(name = "b")
y_ = ad.Variable(name = "y_")

y = ad.matmul_op(x,W) + b

loss = ad.softmaxcrossentropy_op(y, y_)

grad_W, grad_b = ad.gradients(loss, [W, b])

executor = ad.Executor([loss, grad_W, grad_b])

lr1 = 1e-8
lr2 = 1e-8

Ws = np.zeros([784, 10])
bs = np.zeros([10])

print("Begin Training")
for i in range(10000): #1000
    slice = random.sample(image, 100)
    batch_x = []
    batch_y = []
    for j in range(100):
        batch_x.append(slice[j][0])
        batch_y.append(slice[j][1])
    batch_xs = np.array(batch_x)
    batch_ys = np.array(batch_y)
    #sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
    losss, grad_Ws, grad_bs = executor.run(feed_dict = {
        x: batch_xs,
        W: Ws,
        b: bs,
        y_: batch_ys
    })
    Ws -= grad_Ws * lr1
    bs -= np.mean(grad_bs, 0) * lr2
    if i%100 == 0: print(losss)

test_xs = []
for i in timage:
    test_xs.append(np.array(i))
test_xs = np.array(test_xs)

test_ys = []
for i in tlabel:
    temp = np.zeros(10)
    temp[i] = 1
    test_ys.append(temp)
test_ys = np.array(test_ys)

tester = ad.Executor([y])

#test accuracy's calculation
#Ws = np.ones([784, 10])
#bs = np.ones([10])
#oh yeah it's correct...

ys, = tester.run(feed_dict = {
    x: test_xs,
    W: Ws,
    b: bs
})

np.set_printoptions(threshold=np.NaN)
print("Ws:")
print(Ws)
print("bs:")
print(bs)

correct_prediction = np.equal(np.argmax(ys,1), np.argmax(test_ys,1))
accuracy = np.mean(correct_prediction)
print("accuracy : ")
print(accuracy)
