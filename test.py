import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Lambda, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

# データセットの準備
(x_train, _), (x_test, _) = mnist.load_data()
image_size = x_train.shape[1]

#入力画像の中身可視化
plt.figure(figsize=(20, 4))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Image {i}')
    plt.axis('off')  # 軸の目盛りを表示しない
plt.tight_layout()
plt.show()

#元々(60000, 28, 28)の3次元テンソル→(60000, 28, 28, 1)の4次元テンソルに変換
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

#各ピクセルの値は0から255の整数値で表現/0は黒、255は白を意味します。
#ピクセル値を0から1の範囲に正規化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# VAEモデルのパラメータ
#MNIST 画像は 28x28 ピクセルのグレースケール (1チャンネル) なので、入力サイズは (28, 28, 1) となる
input_shape = (image_size, image_size, 1)
batch_size = 128
latent_dim = 2
epochs = 10

# エンコーダの構築
inputs = Input(shape=input_shape, name='encoder_input')#Input関数でモデルの入力層を定義
x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)#Conv2D は2次元の畳み込み層で、画像データの特徴抽出/カーネルサイズは 3x3
x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = Flatten()(x)
#Dense(全結合層)で16のユニットを持つ/潜在空間のパラメータを生成するための中間層
x = Dense(16, activation='relu')(x)
#全結合層の出力値(平均と対数分散)の定義
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# サンプリング関数

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))#標準正規分布
    return z_mean + K.exp(0.5 * z_log_var) * epsilon #returnでリパラ

#z_mean と z_log_var を入力として受け取り、定義した"サンプリング関数"を通じて潜在変数 z を生成
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
#Model関数で入力から出力までの計算をカプセル化するモデルオブジェクトを定義←つまりエンコーダのモデル定義
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

#潜在変数のサンプリングの視覚化
z_mean_pred, z_log_var_pred, z_pred = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(z_pred[:, 0], z_pred[:, 1], alpha=0.5)
plt.title('Visualization of Latent Variables $z$')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.grid(True)

# デコーダの構築
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(7 * 7 * 64, activation='relu')(latent_inputs)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
decoder_outputs = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
decoder = Model(latent_inputs, decoder_outputs, name='decoder')

# VAEモデルのインスタンス化
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# 損失関数の定義
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# モデルのトレーニング
# モデルのトレーニング
vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

# モデルのトレーニングを完了
vae.fit(x_train, epochs=epochs, batch_size=batch_size)

import matplotlib.pyplot as plt

def generate_images(decoder, latent_dim, num_images):
    # ランダムな潜在ベクトルを生成
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))

    # デコーダを使って画像を生成
    generated_images = decoder.predict(random_latent_vectors)
    return generated_images

# 生成する画像の数
num_images = 20


# 画像の生成
generated_images = generate_images(decoder, latent_dim, num_images)

# 生成した画像を表示
plt.figure(figsize=(20, 4))
for i in range(num_images):
    ax = plt.subplot(2, num_images, i + 1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()
