import nose
import mock
import numpy as np
import pandas as pd
import random
from neurons.hmc import HMCNeuron
from neurons.util import zero_center, normalize


class TestCaseHMCNeuronActualExample:

    def test_hmc_neuron_full_example(self):
        """
        A complete HMCNeuron example:

        Create a classification problem by initiating 0,1 class labels by
        sampling 100 data points from a binomial distribution.

        First, create two predictor inputs; Artificially simulate that one
        predicts well and the other doesn't. Then, zero center and normalize
        the inputs and initialize the neuron. Finally, integrate the converged
        weights and make predictions.

        """
        actual_outputs = np.random.binomial(1, .3, 100)
        df = pd.DataFrame({'output':actual_outputs})
        df['input_1'] = np.ones(df.shape[0])
        df['input_2'] = np.ones(df.shape[0])
        feat_cols = ['input_1','input_2']
        # Make 'input_1' a good predictor
        df.ix[df['output']!=1, 'input_1'] = 0
        # Make 'input_2' a noisy, random, bad predictor
        df.ix[df.sample(30).index, 'input_2'] = 0
        df = zero_center(df, 'input_1')
        df = zero_center(df, 'input_2')
        df = normalize(df, 'input_1')
        df = normalize(df, 'input_2')
        hmc = HMCNeuron(df[feat_cols],df['output'],iterations=100)
        hmc.average_converged_weights()
        df['predictions'] = df[feat_cols].apply(
            lambda row: hmc.predict_row(row), axis=1
        )
        print(hmc.weights)
        print('doesnt',df[df['predictions']!= df['output']])
        print('does',df[df['predictions']== df['output']])
