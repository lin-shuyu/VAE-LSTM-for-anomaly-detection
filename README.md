# VAE-LSTM-for-anomaly-detection

This Github repository hosts our code to train a VAE-LSTM hybrid model for anomaly detection. In short, our anomaly detection model contains:
  * a VAE unit which summarizes the local information of a short window into a low-dimensional embedding,
  * a LSTM model, which acts on the low- dimensional embeddings produced by the VAE model, to manage the sequential patterns over longer term.

An overview of our model is shown below:
![System sketch](/figures/detailed_architecture.png)
