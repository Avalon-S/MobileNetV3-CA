import tensorflow as tf
from keras.layers import Lambda,Concatenate,Reshape,Conv2D,BatchNormalization,Activation,Multiply,Add

def coordinate(inputs,ratio=2, name="name"):
    W,H,C = [int(x) for x in inputs.shape[1:]]
    temp_dim = max(int(C//ratio),ratio)
    H_pool = Lambda(lambda x: tf.reduce_mean(x, axis=1))(inputs)
    W_pool = Lambda(lambda x: tf.reduce_mean(x, axis=2))(inputs)
    x = Concatenate(axis=1)([H_pool,W_pool])
    x = Reshape((1,W+H,C))(x)
    x = Conv2D(temp_dim,1, name=name+'1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_h,x_w = Lambda(lambda x:tf.split(x,[H,W],axis=2))(x)
    x_w = Reshape((W,1,temp_dim))(x_w)

    x_h = Conv2D(C,1,activation='sigmoid',name=name+"2")(x_h)
    x_w = Conv2D(C, 1, activation='sigmoid',name=name+"3")(x_w)
    x = Multiply()([inputs,x_h,x_w])
    x = Add()([inputs,x])
    return x