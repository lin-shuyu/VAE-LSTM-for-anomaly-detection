""" util function.py """

import json
import os
import argparse
import tensorflow as tf
from datetime import datetime


def get_config_from_json(json_file):
  """
  Get the config from a json file
  :param json_file:
  :return: config(dictionary)
  """
  # parse the configurations from the config json file provided
  with open(json_file, 'r') as config_file:
    config_dict = json.load(config_file)

  return config_dict


def save_config(config):
  dateTimeObj = datetime.now()
  timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")
  filename = config['result_dir'] + 'training_config_{}.txt'.format(timestampStr)
  config_to_save = json.dumps(config)
  f = open(filename, "w")
  f.write(config_to_save)
  f.close()


def process_config(json_file):
  config = get_config_from_json(json_file)

  # create directories to save experiment results and trained models
  if config['load_dir'] == "default":
    save_dir = "../experiments/local-results/{}/{}/batch-{}".format(
      config['exp_name'], config['dataset'], config['batch_size'])
  else:
    save_dir = config['load_dir']
  # specify the saving folder name for this experiment
  if config['TRAIN_sigma'] == 1:
    save_name = '{}-{}-{}-{}-{}-trainSigma'.format(config['exp_name'],
                                                   config['dataset'],
                                                   config['l_win'],
                                                   config['l_seq'],
                                                   config['code_size'])
  else:
    save_name = '{}-{}-{}-{}-{}-fixedSigma-{}'.format(config['exp_name'],
                                                      config['dataset'],
                                                      config['l_win'],
                                                      config['l_seq'],
                                                      config['code_size'],
                                                      config['sigma'])
  config['summary_dir'] = os.path.join(save_dir, save_name, "summary/")
  config['result_dir'] = os.path.join(save_dir, save_name, "result/")
  config['checkpoint_dir'] = os.path.join(save_dir, save_name, "checkpoint/")
  config['checkpoint_dir_lstm'] = os.path.join(save_dir, save_name, "checkpoint/lstm/")

  return config


def create_dirs(dirs):
  """
  dirs - a list of directories to create if these directories are not found
  :param dirs:
  :return exit_code: 0:success -1:failed
  """
  try:
    for dir_ in dirs:
      if not os.path.exists(dir_):
        os.makedirs(dir_)
    return 0
  except Exception as err:
    print("Creating directories error: {0}".format(err))
    exit(-1)


def count_trainable_variables(scope_name):
  total_parameters = 0
  for variable in tf.trainable_variables(scope_name):
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
      variable_parameters *= dim.value
    total_parameters += variable_parameters
  print(
    'The total number of trainable parameters in the {} model is: {}'.format(scope_name, total_parameters))
  return total_parameters


def get_args():
  argparser = argparse.ArgumentParser(description=__doc__)
  argparser.add_argument(
    '-c', '--config',
    metavar='C',
    default='None',
    help='The Configuration file')
  args = argparser.parse_args()
  return args
