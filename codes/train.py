import os
import tensorflow as tf
from data_loader import DataGenerator
from models import VAEmodel, lstmKerasModel
from trainers import vaeTrainer
from utils import process_config, create_dirs, get_args, save_config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    # save the config in a txt file
    save_config(config)
    # create tensorflow session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # create your data generator
    data = DataGenerator(config)
    # create a CNN model
    model_vae = VAEmodel(config)
    # create a trainer for VAE model
    trainer_vae = vaeTrainer(sess, model_vae, data, config)
    model_vae.load(sess)
    # here you train your model
    if config['TRAIN_VAE']:
        if config['num_epochs_vae'] > 0:
            trainer_vae.train()

    if config['TRAIN_LSTM']:
        # create a lstm model class instance
        lstm_model = lstmKerasModel(data)

        # produce the embedding of all sequences for training of lstm model
        # process the windows in sequence to get their VAE embeddings
        lstm_model.produce_embeddings(config, model_vae, data, sess)

        # Create a basic model instance
        lstm_nn_model = lstm_model.create_lstm_model(config)
        lstm_nn_model.summary()   # Display the model's architecture
        # checkpoint path
        checkpoint_path = config['checkpoint_dir_lstm']\
                          + "cp.ckpt"
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        # load weights if possible
        lstm_model.load_model(lstm_nn_model, config, checkpoint_path)

        # start training
        if config['num_epochs_lstm'] > 0:
            lstm_model.train(config, lstm_nn_model, cp_callback)

        # make a prediction on the test set using the trained model
        lstm_embedding = lstm_nn_model.predict(lstm_model.x_test, batch_size=config['batch_size_lstm'])
        print(lstm_embedding.shape)

        # visualise the first 10 test sequences
        for i in range(10):
            lstm_model.plot_lstm_embedding_prediction(i, config, model_vae, sess, data, lstm_embedding)


if __name__ == '__main__':
    main()
