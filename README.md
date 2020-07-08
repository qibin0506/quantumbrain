# quantumbrain

## Train example

``` python
import quantumbrain as qb
from quantumbrain.datasets import mnist
import numpy as np

# resnet block
def res_block(name, inputs, filters, down_sample=False):
    strides = 2 if down_sample else 1

    layer = qb.layers.Conv(filters, 1, 1, kernel_initializer="he", name=name+"conv:1")(inputs)
    layer = qb.layers.Relu(name=name+"relu:1")(layer)

    layer = qb.layers.Conv(filters, 3, strides, kernel_initializer="he", name=name+"conv:2")(layer)
    layer = qb.layers.Relu(name=name+"relu:2")(layer)

    shortcut = inputs
    if down_sample:
        shortcut = qb.layers.Conv(filters, 1, strides, name=name+"conv:3")(shortcut)

    layer = qb.layers.Add(name=name+"add:1")([layer, shortcut])
    layer = qb.layers.Relu(name=name+"relu:3")(layer)

    return layer


def create_model():
    inputs = qb.layers.Input([None, 1, 28, 28])
    x = qb.layers.Conv(6, 3, 2, kernel_initializer="he")(inputs)
    x = qb.layers.Relu()(x)

    x = res_block("block1_0", x, 6, True)
    for idx in range(2):
        x = res_block("block1_{}".format(1 + idx), x, 6, False)

    x = qb.layers.Dropout(drop_rate=0.5)(x)
    x = qb.layers.Flatten()(x)
    x = qb.layers.Dense(100, kernel_initializer="he")(x)
    x = qb.layers.Relu()(x)
    x = qb.layers.Dense(50, kernel_initializer="he")(x)
    x = qb.layers.Relu()(x)
    output = qb.layers.Dense(10)(x)

    model = qb.model.Model(inputs=inputs, outputs=output)
    return model


def accuracy(x, y):
    model.trainable = False
    y_pred = model(x)
    y_pred = np.argmax(y_pred, axis=1)

    if y.ndim != 1:
        y = np.argmax(y, axis=1)

    return np.sum(y_pred == y) / x.shape[0]


model = create_model()
model.summary()
loss = qb.losses.CE()
optimizer = qb.optimizers.Adam()

trainer = qb.trainer.Trainer(model, loss, optimizer)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
batch_size = 100
for epoch in range(10):
    for batch in range(0, len(x_train), batch_size):
        x_batch = x_train[batch:batch+batch_size]
        y_batch = y_train[batch:batch+batch_size]

        pred_value, loss_value = trainer.train(x_batch, y_batch)

    acc = accuracy(x_test, y_test)
    print("epoch: {}, accuracy: {}".format(epoch, acc))

    if epoch == 9:
     model.save("./result/ckpt_{}".format(epoch))
```

Output:


Model: "model"

\----------------------------------------------------------

Layer(type)                   		Output Shape                  

\==========================================================

Input_0(Input)                		[None, 1, 28, 28]

\---------------------------------------------------------

Conv_1(Conv)                  		[None, 6, 14, 14]

\---------------------------------------------------------

Relu_2(Relu)                  		[None, 6, 14, 14]

\---------------------------------------------------------
block1_0conv:1(Conv)          		[None, 6, 14, 14]

\---------------------------------------------------------

block1_0relu:1(Relu)          		[None, 6, 14, 14]

\---------------------------------------------------------

block1_0conv:2(Conv)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_0relu:2(Relu)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_0conv:3(Conv)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_0add:1(Add)            		[None, 6, 7, 7]

\---------------------------------------------------------

block1_0relu:3(Relu)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_1conv:1(Conv)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_1relu:1(Relu)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_1conv:2(Conv)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_1relu:2(Relu)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_1add:1(Add)            		[None, 6, 7, 7]

\---------------------------------------------------------

block1_1relu:3(Relu)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_2conv:1(Conv)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_2relu:1(Relu)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_2conv:2(Conv)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_2relu:2(Relu)          		[None, 6, 7, 7]

\---------------------------------------------------------

block1_2add:1(Add)            		[None, 6, 7, 7]

\---------------------------------------------------------

block1_2relu:3(Relu)          		[None, 6, 7, 7]

\---------------------------------------------------------

Dropout_22(Dropout)           		[None, 6, 7, 7]

\---------------------------------------------------------

Flatten_23(Flatten)           		[None, 294]

\---------------------------------------------------------

Dense_24(Dense)               		[None, 100]

\---------------------------------------------------------

Relu_25(Relu)                 		[None, 100]

\---------------------------------------------------------

Dense_26(Dense)               		[None, 50]

\---------------------------------------------------------

Relu_27(Relu)                 		[None, 50]

\---------------------------------------------------------

Dense_28(Dense)               		[None, 10]

\==========================================================

epoch: 0, accuracy: 0.8924

epoch: 1, accuracy: 0.9162

epoch: 2, accuracy: 0.9378

epoch: 3, accuracy: 0.9417

epoch: 4, accuracy: 0.9465

epoch: 5, accuracy: 0.9531

epoch: 6, accuracy: 0.9513

epoch: 7, accuracy: 0.9554

epoch: 8, accuracy: 0.9568

epoch: 9, accuracy: 0.9595

## save & restore weights

``` python
# save
model.save("./result/ckpt_{}".format(epoch))

# restore
model = create_model()
model.restore("./result/ckpt_9")
```

## predict

``` python
model = create_model()
model.restore("./result/ckpt_9")
y_pred = model(x)
y_pred = qb.softmax(y_pred)
print(y_pred)
```
