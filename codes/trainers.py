from base import BaseTrain
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from scipy.stats import multivariate_normal


class vaeTrainer(BaseTrain):
  def __init__(self, sess, model, data, config):
    super(vaeTrainer, self).__init__(sess, model, data, config)

  def train_epoch(self):
    self.cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)

    # training
    self.sess.run(self.model.iterator.initializer,
                  feed_dict={self.model.original_signal: self.data.train_set_vae['data'],
                             self.model.seed: self.cur_epoch})
    self.n_train_iter = self.data.n_train_vae // self.config['batch_size']
    idx_check_point = (self.n_train_iter - 1)
    train_loss_cur_epoch = 0.0
    for i in range(self.n_train_iter):
      loss = self.train_step()
      self.sess.run(self.model.increment_global_step_tensor)
      self.train_loss.append(np.squeeze(loss))
      train_loss_cur_epoch = train_loss_cur_epoch + loss
      if i == idx_check_point:
        test_loss, test_recons_loss_weighted, test_kl, test_sigma_regularisor, test_code_std_norm, test_cur_sigma2, test_recons_loss_ls = self.test_step()
    self.train_loss_ave_epoch.append(train_loss_cur_epoch / self.n_train_iter)

    # validation
    self.iter_epochs_list.append(self.n_train_iter * (self.cur_epoch + 1))
    self.sess.run(self.model.iterator.initializer,
                  feed_dict={self.model.original_signal: self.data.val_set_vae['data'],
                             self.model.seed: self.cur_epoch})
    self.n_val_iter = self.data.n_val_vae // self.config['batch_size']
    val_loss_cur_epoch = 0.0
    for i in range(self.n_val_iter):
      val_loss = self.val_step()
      val_loss_cur_epoch = val_loss_cur_epoch + val_loss
    self.val_loss_ave_epoch.append(val_loss_cur_epoch / self.n_val_iter)

    # save the model parameters at the end of this epoch
    self.model.save(self.sess)

    print(
      "{}/{}, test loss: -elbo: {:.4f}, recons_loss_weighted: {:.4f}, recons_loss_ls: {:.4f}, KL_loss: {:.4f}, sigma_regularisor: {:.4f}, code_std_dev: {}".format(
        self.cur_epoch,
        self.config['num_epochs_vae'] - 1,
        test_loss,
        test_recons_loss_weighted,
        np.squeeze(np.mean(test_recons_loss_ls)),
        test_kl,
        test_sigma_regularisor,
        np.squeeze(test_code_std_norm)))
    print("Loss on training and val sets:\ntrain: {:.4f}, val: {:.4f}".format(
      self.train_loss_ave_epoch[self.cur_epoch],
      self.val_loss_ave_epoch[self.cur_epoch]))
    print("Current sigma2: {:.7f}".format(test_cur_sigma2))

    # save the current variables
    self.save_variables_VAE()

    # reconstruction plot
    self.plot_reconstructed_signal()

    # generate samples from prior
    self.generate_samples_from_prior()

    # plot the training and validation loss over iterations/epochs
    self.plot_train_and_val_loss()

  def train_step(self):
    batch_image = self.sess.run(self.model.input_image)
    feed_dict = {self.model.original_signal: batch_image,
                 self.model.is_code_input: False,
                 self.model.code_input: np.zeros((1, self.config['code_size'])),
                 self.model.lr: self.config['learning_rate_vae'] * (0.98 ** self.cur_epoch)}
    train_loss, _ = self.sess.run([self.model.elbo_loss, self.model.train_step_gradient],
                                  feed_dict=feed_dict)
    return train_loss

  def val_step(self):
    input_image_val = self.sess.run(self.model.input_image)
    val_cost, recon_loss_val, kl_loss_val, std_dev_loss_val = self.sess.run([self.model.elbo_loss,
                                                                             self.model.ls_reconstruction_error,
                                                                             self.model.KL_loss,
                                                                             self.model.std_dev_norm],
                                                                            feed_dict={
                                                                              self.model.original_signal: input_image_val,
                                                                              self.model.is_code_input: False,
                                                                              self.model.code_input: np.zeros(
                                                                                (1, self.config['code_size']))})
    self.val_loss.append(np.squeeze(val_cost))
    self.recons_loss_val.append(np.squeeze(np.mean(recon_loss_val)))
    self.KL_loss_val.append(kl_loss_val)
    return val_cost

  def test_step(self):
    feed_dict = {self.model.original_signal: self.data.test_set_vae['data'],
                 self.model.is_code_input: False,
                 self.model.code_input: np.zeros((1, self.config['code_size']))}
    self.output_test, test_loss, test_recons_loss_weighted, test_kl, test_sigma_regularisor, test_code_std_norm, test_cur_sigma2, test_recons_loss_ls = self.sess.run(
      [self.model.decoded,
       self.model.elbo_loss,
       self.model.weighted_reconstruction_error_dataset,
       self.model.KL_loss,
       self.model.sigma_regularisor_dataset,
       self.model.std_dev_norm,
       self.model.sigma2,
       self.model.ls_reconstruction_error],
      feed_dict=feed_dict)
    self.test_sigma2.append(np.squeeze(test_cur_sigma2))
    return test_loss, test_recons_loss_weighted, test_kl, test_sigma_regularisor, test_code_std_norm, np.squeeze(
      test_cur_sigma2), test_recons_loss_ls

  def plot_reconstructed_signal(self):
    input_images = np.squeeze(self.data.test_set_vae['data'])
    decoded_images = np.squeeze(self.output_test)
    n_images = 20
    # plot the reconstructed image for a shape
    for j in range(self.config['n_channel']):
      fig, axs = plt.subplots(4, 5, figsize=(18, 10), edgecolor='k')
      fig.subplots_adjust(hspace=.4, wspace=.4)
      axs = axs.ravel()
      for i in range(n_images):
        if self.config['n_channel'] == 1:
          axs[i].plot(input_images[i])
          axs[i].plot(decoded_images[i])
        else:
          axs[i].plot(input_images[i, :, j])
          axs[i].plot(decoded_images[i, :, j])
        axs[i].grid(True)
        axs[i].set_xlim(0, self.config['l_win'])
        axs[i].set_ylim(-5, 5)
        if i == 19:
          axs[i].legend(('original', 'reconstructed'))
      plt.suptitle('Channel {}'.format(j))
      savefig(self.config['result_dir'] + 'test_reconstructed_{}_{}.pdf'.format(self.cur_epoch, j))
      fig.clf()
      plt.close()

  def generate_samples_from_prior(self):
    rv = multivariate_normal(np.zeros(self.config['code_size']), np.diag(np.ones(self.config['code_size'])))
    # Generate a batch size of samples from the prior samples
    n_images = 20
    samples_code_prior = rv.rvs(n_images)
    sampled_images = self.sess.run(self.model.decoded,
                                   feed_dict={self.model.original_signal: np.zeros(
                                     (n_images, self.config['l_win'], self.config['n_channel'])),
                                              self.model.is_code_input: True,
                                              self.model.code_input: samples_code_prior})
    sampled_images = np.squeeze(sampled_images)
    for j in range(self.config['n_channel']):
      fig, axs = plt.subplots(4, 5, figsize=(18, 10), edgecolor='k')
      fig.subplots_adjust(hspace=.4, wspace=.4)
      axs = axs.ravel()
      for i in range(n_images):
        if self.config['n_channel'] == 1:
          axs[i].plot(sampled_images[i])
        else:
          axs[i].plot(sampled_images[i, :, j])
        axs[i].grid(True)
        axs[i].set_xlim(0, self.config['l_win'])
        axs[i].set_ylim(-5, 5)
      plt.suptitle('Channel {}'.format(j))
      savefig(self.config['result_dir'] + 'generated_samples_{}_{}.pdf'.format(self.cur_epoch, j))
      fig.clf()
      plt.close()
