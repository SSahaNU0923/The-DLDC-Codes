from __INCLUDE__ import *

def custom_loss(y_true, y_pred):
    system_internal_energy = tf.math.cumsum(y_pred, axis=0)
    # print('Internal energy:' + str(system_internal_energy))
    external_work = tf.math.reduce_mean(y_true, axis=0)
    # print('External work:' + str(external_work))
    loss = tf.keras.metrics.mean_absolute_error(external_work, system_internal_energy)
    return loss

def load_data(folder):
    os.chdir(folder)
    matfiles = glob.glob("*.mat")
    datalist = []
    for filename in matfiles:
        dtemp = scipy.io.loadmat(filename)
        print(dtemp)
        datalist.append(dtemp['Z'])
    data = np.array(datalist)
    os.chdir('..')
    return data


# Loading and training
if __name__ == '__main__':
    #Load data
    folder_name = 'data2/epsx'
    y = load_data(folder_name)

    y = y[:, 1:, 1:]
    # Strain applied
    eps_max = 0.01
    eps = np.linspace(0.0001, eps_max, y.shape[0])
    print(eps)

    # Compute eigenstrain
    yeig = y.copy()
    for i in range(y.shape[0]):
        yeig[i, :, :] = y[i, :, :] - eps[i]

    ## Material properties
    Em, nim = 3.79, .39
    E1f, E2f, E3f, nif = 245, 19.8, 19.8, 0.671

    xscaler = StandardScaler()
    x_train_pre = y[:95, :, :]
    x_train = xscaler.fit_transform(x_train_pre.reshape(x_train_pre.shape[0], -1)).reshape(x_train_pre.shape)
    x_test_pre = y[95:, :, :]
    x_test = xscaler.transform(x_test_pre.reshape(x_test_pre.shape[0], -1)).reshape(x_test_pre.shape)

    yscaler = StandardScaler()
    y_train_pre = yeig[:95, :, :]
    y_train = yscaler.fit_transform(y_train_pre.reshape(y_train_pre.shape[0], -1)).reshape(y_train_pre.shape)
    y_test_pre = yeig[95:, :, :]
    y_test = yscaler.transform(y_test_pre.reshape(y_test_pre.shape[0], -1)).reshape(y_test_pre.shape)

    # Show pic
    n = 3  # how many records we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_pre[i].reshape(90, 90))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    kernel_size = (3, 3)
    pool_size = (2, 2)
    # MODEL
    conv_inp = Input(shape=(90, 90, 1), name='conv_inp')
    # conv_inp = tf.keras.layers.Cropping2D(cropping=((1, 0), (1, 0)))(conv_inp_raw)

    # Conv1 #
    x = Conv2D(filters=64, kernel_size=kernel_size, activation='relu', padding='same')(conv_inp)
    x = MaxPooling2D(pool_size=pool_size, padding='same')(x)

    # Conv2 #
    x = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)

    # Conv 3 #
    x = Conv2D(filters=16, kernel_size=kernel_size, activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(3, 3), padding='same')(x)

    # DeConv1
    x = Conv2DTranspose(64, kernel_size, strides=3, activation='relu', padding='same')(encoded)
    # x = UpSampling2D((3, 3))(x)

    # DeConv2
    x = Conv2DTranspose(32, kernel_size, strides=3, activation='relu', padding='same')(x)
    # x = UpSampling2D(pool_size)(x)

    # Deconv3
    x = Conv2DTranspose(16, kernel_size, strides=2, padding='same', activation='relu')(x)
    # x = UpSampling2D(pool_size)(x)


    decoded = Conv2D(1, kernel_size, activation='sigmoid', padding='same')(x)

    # Declare the model
    autoencoder = Model(conv_inp, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.summary()
    # Train the model
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    history= autoencoder.fit(x_train, x_train,
                            epochs=100,
                            batch_size=128,
                            shuffle=True,
                            validation_data=(x_test, y_test),
                             callbacks=early_stop
                            )

    decoded_imgs = autoencoder.predict(x_test)
    decoded_imgs = yscaler.inverse_transform(decoded_imgs.reshape(decoded_imgs.shape[0], -1)).reshape(decoded_imgs.shape)

    n = 3
    fig = plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x_test_pre[i].reshape(90, 90))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar()

    fig = plt.figure(figsize=(20, 4))
    for i in range(n):
        # display reconstruction
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(decoded_imgs[i].reshape(90, 90))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar()

    plt.show()

    etest = 0.01
    e0 = etest*np.ones((90, 90))
    e0 = e0.reshape(1, e0.shape[0], e0.shape[1], 1)
    ei = e0
    ei_list = []
    ei_eig_list = []
    err2d_list = []
    err1d_list = []
    for it in range(100):
        ei_list.append(ei)
        ei_scaled = xscaler.transform(ei.reshape(1, -1)).reshape(ei.shape)
        ei_eig_sc = autoencoder.predict(ei_scaled)
        ei_eig = yscaler.inverse_transform(ei_eig_sc.reshape(1, -1)).reshape(ei_eig_sc.shape)
        ei = ei_eig - ei
        ei_eig_list.append(ei_eig)
        ae_err = ei.reshape(90, 90) - y[-1, :, :]
        err2d_list.append(ae_err)
        err1d_list.append(mse(y[-1, :, :].reshape(-1, 1), ei.reshape(-1, 1)))

    n = 10
    fig = plt.figure(figsize=(20, 4))
    idx = range(1, 100, 10)
    for i in range(n):
        j = idx[i]
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(ei_list[i].reshape(90, 90))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar()

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(ei_eig_list[i].reshape(90, 90))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar()

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        err = np.abs(ei_list[i].reshape(90, 90) - y[-1, :, :])
        if i==0:
            im = plt.imshow(err.reshape(90, 90))
        else:
            plt.imshow(err.reshape(90, 90))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar(im)

    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Training')
    # plt.plot(history.history['val_loss'], label='Validation [5%]')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.show()

    plt.figure()
    plt.plot(np.array(err1d_list))
    plt.show()