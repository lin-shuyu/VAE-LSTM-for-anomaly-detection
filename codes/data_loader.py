from base import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig


class DataGenerator(BaseDataGenerator):
  def __init__(self, config):
    super(DataGenerator, self).__init__(config)
    # load data here: generate 3 state variables: train_set, val_set and test_set
    self.load_NAB_dataset(self.config['dataset'], self.config['y_scale'])

  def load_NAB_dataset(self, dataset, y_scale=6):
    data_dir = '../datasets/NAB-known-anomaly/'
    data = np.load(data_dir + dataset + '.npz')

    # normalise the dataset by training set mean and std
    train_m = data['train_m']
    train_std = data['train_std']
    readings_normalised = (data['readings'] - train_m) / train_std

    # plot normalised data
    fig, axs = plt.subplots(1, 1, figsize=(18, 4), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs.plot(data['t'], readings_normalised)
    if data['idx_split'][0] == 0:
      axs.plot(data['idx_split'][1] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    else:
      for i in range(2):
        axs.plot(data['idx_split'][i] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    axs.plot(*np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b--')
    for j in range(len(data['idx_anomaly'])):
      axs.plot(data['idx_anomaly'][j] * np.ones(20), np.linspace(-y_scale, 0.75 * y_scale, 20), 'r--')
    axs.grid(True)
    axs.set_xlim(0, len(data['t']))
    axs.set_ylim(-y_scale, y_scale)
    axs.set_xlabel("timestamp (every {})".format(data['t_unit']))
    axs.set_ylabel("readings")
    axs.set_title("{} dataset\n(normalised by train mean {:.4f} and std {:.4f})".format(dataset, train_m, train_std))
    axs.legend(('data', 'train test set split', 'anomalies'))
    savefig(self.config['result_dir'] + '/raw_data_normalised.pdf')

    # slice training set into rolling windows
    n_train_sample = len(data['training'])
    n_train_vae = n_train_sample - self.config['l_win'] + 1
    rolling_windows = np.zeros((n_train_vae, self.config['l_win']))
    for i in range(n_train_sample - self.config['l_win'] + 1):
      rolling_windows[i] = data['training'][i:i + self.config['l_win']]

    # create VAE training and validation set
    idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
    self.train_set_vae = dict(data=np.expand_dims(rolling_windows[idx_train], -1))
    self.val_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val], -1))
    self.test_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val[:self.config['batch_size']]], -1))

    # create LSTM training and validation set
    for k in range(self.config['l_win']):
      n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
      n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
      cur_lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win']))
      for i in range(n_train_lstm):
        cur_seq = np.zeros((self.config['l_seq'], self.config['l_win']))
        for j in range(self.config['l_seq']):
          # print(k,i,j)
          cur_seq[j] = data['training'][k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]
        cur_lstm_seq[i] = cur_seq
      if k == 0:
        lstm_seq = cur_lstm_seq
      else:
        lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)

    n_train_lstm = lstm_seq.shape[0]
    idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
    self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
    self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))

  def plot_time_series(self, data, time, data_list):
    fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
    fig.subplots_adjust(hspace=.8, wspace=.4)
    axs = axs.ravel()
    for i in range(4):
      axs[i].plot(time / 60., data[:, i])
      axs[i].set_title(data_list[i])
      axs[i].set_xlabel('time (h)')
      axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
    savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')
