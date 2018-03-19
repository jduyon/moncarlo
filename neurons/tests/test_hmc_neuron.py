import nose
import mock
import numpy as np
import pandas as pd
import random
from neurons.hmc import HMCNeuron
from neurons.util import zero_center, normalize

class TestCaseHMCNeuronInit:

    def test_hmc_neuron_init_takes_three_args(self):
        neuron = HMCNeuron('inputs','outputs','iterations')

    @nose.tools.raises(TypeError)
    def test_hmc_neuron_init_raises_when_no_args(self):
        neuron = HMCNeuron()

    def test_hmc_neuron_init_stores_arguments(self):
        """
        Test that inputs, outputs and iterations get set to the instance.
        """
        neuron = HMCNeuron('inputs','outputs','iterations')
        assert(neuron.inputs == 'inputs')
        assert(neuron.outputs == 'outputs')
        assert(neuron.iterations == 'iterations')

class TestCaseHMCNeuronWeights:

    def test_store_weights_stores_weights(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        weights = [1,2,3]
        hmc.store_weights(weights)
        assert(hmc.weight_samples == [[1,2,3]])

    @nose.tools.raises(TypeError)
    def test_store_weights_raises_when_no_args(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.store_weights()

    @nose.tools.raises(TypeError)
    def test_store_weights_raises_when_too_many_args(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.store_weights(1,2,3)

    def test_init_weights_shape_is_correct(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        weight_count = 3
        weights = hmc.init_weights(weight_count)
        actual = weights.shape[0]
        expected = weight_count
        assert(actual == expected)

    @nose.tools.raises(TypeError)
    def test_init_weights_shape_raises_when_no_args_raises(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.init_weights()

    @nose.tools.raises(TypeError)
    def test_init_weights_shape_raises_when_too_many_args(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.init_weights(3,2)

class TestCaseHMCNeuronMomentum:
    def test_init_momentum_shape_is_correct(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        columns = 3
        momentum = hmc.init_momentum(columns)
        actual = momentum.shape[0]
        expected = columns
        assert(actual == expected)

    @nose.tools.raises(TypeError)
    def test_init_momentum_shape_raises_when_no_args_raises(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.init_momentum()

    @nose.tools.raises(TypeError)
    def test_init_momentum_shape_raises_when_too_many_args(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.init_momentum(3,2)

class TestCaseHMCNeuronHamltonianMomentum:
    def test_hamiltonian_grad_calcs_example(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.grad_obj = mock.Mock()
        hmc.grad_obj.return_value = np.array([.02,.02]) # Fake gradient calc
        weights = np.array([.1,.2]) # Fake initialized weights
        momentum = np.array([.01,.02]) # Fake initial momentum
        gradient = np.array([.01,.01]) # Another fake gradient
        momentum = hmc.hamiltonian_grad_calcs(momentum,weights,gradient)

    @nose.tools.raises(TypeError)
    def test_hamiltonian_grad_calcs_not_enough_args(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        momentum = hmc.hamiltonian_grad_calcs()

    @nose.tools.raises(TypeError)
    def test_hamiltonian_grad_calcs_too_many_args(self):
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        momentum = hmc.hamiltonian_grad_calcs([],[],[],[])

    def test_hamiltonian_grad_calcs_args_must_be_np_arrays(self):
        """ When the args aren't numpy arrays, it should fail."""
        with nose.tools.assert_raises(TypeError) as te:
            hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
            hmc.grad_obj = mock.Mock()
            hmc.grad_obj.return_value = [.02,.02]
            weights = np.array([.1,.2])
            momentum = np.array([.01,.02])
            gradient = np.array([.01,.01])
            momentum = hmc.hamiltonian_grad_calcs(momentum,weights,gradient)
        expected = 'can\'t multiply sequence by non-int of type \'float\''
        actual = str(te.exception)
        assert(actual == expected)

    def test_hamiltonian_grad_calcs_return_value_must_be_np_array(self):
        """
        The return value of hamiltonian_grad_calcs should be a numpy array;
        If this were simply an array, a type error would be raised once I
        multiply momentum by .5 (which is exactly what happens next in the
        training algorithm).
        """
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.grad_obj = mock.Mock()
        hmc.grad_obj.return_value = np.array([.02,.02])
        weights = np.array([.1,.2])
        momentum = np.array([.01,.02])
        gradient = np.array([1,1])
        momentum = hmc.hamiltonian_grad_calcs(momentum,weights,gradient)
        momentum * .5

    def test_hamiltonian_grad_calcs_return_array_correct_dimensions(self):
        """
        The return value of hamiltonian_grad_calcs should be 1 row with
        an element for each input column.
        """
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.grad_obj = mock.Mock()
        hmc.grad_obj.return_value = np.array([.02,.02,.4])
        weights = np.array([.1,.2,.2])
        momentum = np.array([.01,.02,.1])
        gradient = np.array([.01,.01,.1])
        momentum = hmc.hamiltonian_grad_calcs(momentum,weights,gradient)
        actual = len(momentum)
        expected = 3
        assert(actual==expected)

    @mock.patch('hmc.np.random.randint')
    def test_hamiltonian_grad_calcs_randomly_generates_tau(self,mock_randint):
        """
        Make sure the number of gradient calculations is randomly chosen.
        """
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.grad_obj = mock.Mock()
        hmc.grad_obj.return_value = np.array([.02,.02])
        weights = np.array([.1,.2])
        momentum = np.array([.01,.02])
        gradient = np.array([.01,.01])
        momentum = hmc.hamiltonian_grad_calcs(momentum,weights,gradient)
        assert(mock_randint.called_once)

    @mock.patch('hmc.np.random.randint')
    def test_hamiltonian_grad_calcs_must_call_grad_obj_count(self,mock_randint):
        mock_randint.return_value = 100
        hmc = HMCNeuron('fake_arg','fake_arg','fake_arg')
        hmc.grad_obj = mock.Mock()
        hmc.grad_obj.return_value = np.array([.02,.02])
        weights = np.array([.1,.2])
        momentum = np.array([.01,.02])
        gradient = np.array([.01,.01])
        momentum = hmc.hamiltonian_grad_calcs(momentum,weights,gradient)
        actual = hmc.grad_obj.call_count
        expected = 100
        assert(actual == expected)
