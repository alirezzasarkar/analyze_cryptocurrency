# config.py
import os
import logging
import random
import numpy as np
import tensorflow as tf
import plotly.io as pio

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Configure logging
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logging.info('Product execution started')

# Configure Plotly defaults
pio.renderers.default = "colab"
pio.templates.default = "plotly_white"
