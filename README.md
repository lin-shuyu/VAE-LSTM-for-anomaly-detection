# VAE-LSTM-for-anomaly-detection

This Github repository hosts our code and pre-processed data to train a VAE-LSTM hybrid model for anomaly detection, as proposed in our paper Anomaly Detection for Time Series Using VAE-LSTM Hybrid Model at ICASSP 2020. 

In short, our anomaly detection model contains:
  * a VAE unit which summarizes the local information of a short window into a low-dimensional embedding,
  * a LSTM model, which acts on the low- dimensional embeddings produced by the VAE model, to manage the sequential patterns over longer term.

An overview of our model is shown below:

<img align="middle" src="figures/detailed_architecture.png" alt="overview" width="420"/>

An example of anomaly detection on a time series of office temperature, which is provided by Numenta anomaly benchmark (NAB) datasets in their known anomaly subgroup [link](https://github.com/numenta/NAB/tree/master/data/realKnownCause):

<img align="middle" src="figures/ambient_temp_ours.png" alt="result" width="1000"/>

