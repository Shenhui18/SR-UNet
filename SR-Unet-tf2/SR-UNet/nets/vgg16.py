from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from nets.attention import se_block, cbam_block, eca_block,RAM,RRB

attention=[se_block, cbam_block, eca_block,RAM,RRB]


def VGG16(img_input,phi=1):
    # Block 1
    # 512,512,3 -> 512,512,64
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block1_conv2')(x)
    if phi >= 1 and phi <= 3:
        x = attention[phi-1](x, name='x')

    #添加RAM模块
    x = attention[3](x, name='feat5')

    feat1 = x
    # 512,512,64 -> 256,256,64
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # 256,256,64 -> 256,256,128
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block2_conv2')(x)

    if phi >= 1 and phi <= 3:
        x = attention[phi-1](x, name='x2')

    feat2 = x
    # 256,256,128 -> 128,128,128
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # Block 3
    # 128,128,128 -> 128,128,256
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block3_conv3')(x)

    xx = x

    if phi >= 1 and phi <= 3:
        x = attention[phi-1](x, name='x3')



    feat3 = x
    # 128,128,256 -> 64,64,256
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    # 64,64,256 -> 64,64,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block4_conv3')(x)
    if phi >= 1 and phi <= 3:
        x = attention[phi-1](x, name='x4')


    feat4 = x
    # 64,64,512 -> 32,32,512
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # 32,32,512 -> 32,32,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block5_conv3')(x)
    if phi >= 1 and phi <= 3:
        x = attention[phi-1](x, name='x5')


    feat5 = x
    return feat1, feat2, feat3, feat4, feat5 ,xx
