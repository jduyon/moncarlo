import numpy as np

class HMCNeuron(object):
    """

    The classification problem is where you have some input data, and
    you want to find some functional relationship between this data and
    it's underlying classification.

    A Hamiltonian Monte Carlo (HMC) binary classifier is an approach to
    making bayesian predictions for the classification problem. By using
    a gradient descent algorithm with added noise, the HMC neuron is
    able to find samples of weights, from their empirical distribution,
    that show the relationship between input data and their target
    classes. By integrating over these samples (and marginalizing over
    all other parameters), predictions and your belief of those
    predictions can be made.

    [HMC vs Naive gradient descent (ie: steepest descent)]
    A steepest descent algorithm is a function that iteratively
    minimizes some objective function. The objective function in this
    case is the predicted errors. Steepest descent works by calculating
    the gradient of this objective function, and updating the weights
    in the direction of the gradient. Because only the gradient is used
    in updating the weights (in steepest descent), the weights will only
    move in the direction where the objective function (the error) is
    minimized. This causes problems like:

    1. The weights will converge to the most probable, error minimized
       point.
    2. The weights can grow larger or smaller forever creating strict
       decision boundaries.

    Both of these problems lead to over-fitting, which is making over
    confident predictions. A model that over-fits, does not generalize
    well to new data. Also, if we just use the most probable weights
    for our predictions, we leave out information about the distribution
    of the weights. The most probable weights are not representative of
    the typical sample of weights. Wouldn't you want to know what
    portion of the weight distribution classifies a point one way or
    the other? Visually, steepest descent has straight-line contours
    between classifications boundaries. In reality, there are a
    distribution of weights and the contours are less rigid (especially
    where there is a low input sample). Hamiltonian Monte Carlo is able
    to explore this weight distribution and get samples from it by
    employing a random walk strategy and making many gradient
    calculations each step. A step is simply accepting the updated
    weights. By adding a random walk to your gradient descent, you are
    able to explore more than just the most probable weights. Finally,
    by accepting some steps with a random probability, you can ensure
    that detailed balance holds.

    """

    def __init__(self,inputs,outputs,iterations,
        weight_decay=.9,learning_rate=.1,hidden=False):

        """
        Initialize the neuron with model parameters.

        :param inputs: The input DataFrame of a classification problem.
        The index of each input should correspond to the index of the
        output. Each element should be a numerical type.

        :param outputs: The output Series or vector a classification
        problem. The index of each element should correspond to the
        index of the inputs. Each element should be 0 or 1.

        :param iterations: An integer specifying the number of
        iterations of training to run. This doesn't equal the number of
        weight samples to make because a decision rule is enacted each
        round of training.

        :param learning_rate: The hyper parameter which controls how
        big of a step to make during a weight update. Setting this too
        high can result in your weight update "over-stepping"; meaning
        you jump beyond the true weight space. Convergence can take
        longer (or never occur) with a learning rate too large. On the
        other hand, setting this too low will require more iterations
        to converge. A step size too low may prevent the weight update
        from "jumping" out of local-minima weights. Default learning
        rate is .1

        :param weight_decay: The regularization constant hyper
        parameter which modifies the objective function to incorporate a
        bias against larger weights. In the case of a sigmoid activation
        funciton, a large weight decay rate (.9 - 1.0) will result in
        decision boundaries that are closer to the origin. If you don't
        have a weight decay rate or it is too small, this will allow
        the weights to grow as large (or as small) as they can. When you
        activate the inputs * weights to make your predictions, the
        sigmoid will be given much larger or much smaller values. The
        predictions will always be close to 1 or 0, and we lose the
        context of the relative prediction likelihood. Default weight
        decay is .9

        :param hidden: This is used to determine what activation
        function to activate the predictions by. #TODO: Remove this and
        refactor how you decide the activation function. Default is
        False

        :ivar weight_samples: The sampled weights stored from sampling
        the weight space. This includes all weights sampled (from both
        before and after convergence). When inferring the true weights,
        it is best practice to skip the samples before convergence.

        :ivar min_tau: The minimum number of gradient calculations to
        consider per iteration. This is used for implementing the
        "leapfrog" aspect of Hamiltonian Monte Carlo sampling. It should
        be set high enough so to minimize the correlation between
        samples.

        :ivar max_tau: Similar to min_tau; the mid-point of these two is
        how many gradient calculations will be made (on average). If the
        acceptance rate is sufficiently high enough, setting this higher
        unnecessarily slows the algorithm down.
        """

        self.hidden=hidden
        self.inputs = inputs
        self.outputs = outputs
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_samples = []
        self.min_tau = 100
        self.max_tau = 200

    def store_weights(self, weights):
        """
        Store the accepted weight samples.

        :param weights: A weight vector where each element corresponds
        to the input column.
        """
        self.weight_samples.append(weights)

    def init_weights(self, shape):
        """
        Initialize the weights over a uniform distribution centered at
        0.

        :param shape: An integer representing how many weights to
        initialize. Should equal the number of input columns. Returns a
        1 dimensional array.

        :returns: A numpy array of floats.
        """
        return np.random.uniform(-1,1, size=shape)

    def init_momentum(self, shape):
        """
        Initialize the noise vector with samples from a Gaussian
        probability distribution centered at 0. Returns a 1 dimensional
        array.

        :param shape: An integer representing how many momentums to
        initialize. Should equal the number of weights (or input
        columns).

        :returns: A numpy array of floats.
        """
        return np.random.normal(size=shape)

    def leapfrog_method(self, momentum, weights, gradient):
        """
        The hamiltonian can be defined with this equation:

        H(x,p) = E(x) + K(p)

        Where:
            x are the weight samples
            p is momentum
            E(x) is the objective function.

            K(p) is the 'kinetic energy' or momentum function. We use
            this to inform us about which direction to make informed
            steps in weight space towards the minimum of the objective
            function.

            H(x,p) is the Hamiltonian function, and in the context of
            this algorithm, it gives us a point of reference for our
            weights. It is used when comparing and deciding which weight
            samples to accept or reject.

        This algorithm starts by using the initial weights and gradient
        (of their objectve error function) to modify the initial
        momentum. Then, we repeat this process many times so to reduce
        the correlation between the first momentum term and the final.
        Because correlation is reduced between each new weight sample,
        we reject less weight samples. A lower rejection rate means
        that the random walk behavior is minimized as outlined from the
        Metropolis decision rule.

        NOTE:
        If tau is set to 1, this is considered the Langevin method,
        which is when only one step using a gradient calculation is
        used to update the momentum.

        Returns the momentum array, which is 1 dimensional and has one
        element for each weight.

        :param momentum: A numpy array of momentums. Should be 1
        dimensional with one element per weight (or input columns).

        :param weights: A numpy array of weights. Should be 1
        dimensional with one element per input columns.

        :param gradient: A numpy array with a gradient calculation for
        each weight.

        :returns: A numpy array of floats.
        """

        tau = np.random.randint(self.min_tau,self.max_tau)
        for j in range(0,tau):
            momentum = momentum - gradient * self.learning_rate * .5
            new_weights = weights + self.learning_rate * momentum
            new_gradient = self.grad_obj(new_weights)
            momentum = momentum - new_gradient * self.learning_rate *.5
        return momentum

    def grad_obj(self, new_weights):
        #TODO: Implement
        pass

    def find_obj(self, weights):
        #TODO: Implement
        pass

    def sum_hamiltonian(self, momentum, obj_error):
        """
        Calculate the Hamiltonian by summing the objective error and
        the momentum function, K(p). The momentum function is:

        K(p) = p' * p / 2

        Where:
            p is the vector of gaussian samples iteratively updated by
            the gradient of the objective error.
            p' is the above vector transposed.

        The momentum function is half the interation of the guassian
        sample vector with itself or, the dot product. Here's the
        equation for the dot product:

        K(p) = .5 * p[0] * p'[0] + p[1] * p'[1] + p[2] * p'[2] + ...i

        We can calculate this in numpy by using: np.dot(p, p *.5)
        Better momentum functions can be used, and although I call this
        a momentum function, it's name is really the "Kinetic energy".
        Better implementations are demonstrated by Michael Betancourt
        in:

        "A General Metric for Riemannian Manifold Hamiltonian Monte
        Carlo"

        This Hamiltonian function are based on David Mackay's
        implementation form page 495 of:

            "Information Theory, Inference, and Learning Algorithms"

        NOTE:
        In his octave source code, he initializes weights and momentum
        as a column vector, and it's transpose is the row vector. In
        this implementation, the weights and momentum are initialized
        as row vectors, and their transpose are column vectors. In his
        code, he specifically arranges the vectors to get the dot
        product. This is important because if arranged incorrectly, you
        will end up with multi-dimensional matrix instead of one number.

        Returns the sum of: half the dot product of the momentum, and
        the objective error.

        :param momentum: The Monte Carlo sampled momentum vector, one
        momentum term per weight.

        :param obj_error: The sum of the objective error.

        :returns: A float.
        """
        return np.dot(momentum,momentum *.5) + obj_error

    def train(self):
        # TODO: Refactor, get under test.
        weights = []
        # Setup intial weights
        weights = self.init_weights(weights, self.inputs.shape[1])
        # Setup initial gradient and error
        gradient = self.grad_obj(weights)
        obj_error = self.find_obj(weights)
        for i in range(1,self.iterations):
            # Create noise vector for each featcol
            momentum = self.init_momentum(self.inputs.shape[1])
            hamiltonian = self.sum_hamiltonian(momentum, obj_error)
            momentum = self.leapfrog_method(momentum, weights, gradient)
            new_obj_error = self.find_obj(new_weights)
            # We need a way to evaluate a new weight sample. To do this we
            # sum the obj error and the momentum with gradient calculations.
            new_hamiltonian = self.sum_hamiltonian(momentum, obj_error)
            h_diff = new_hamiltonian - hamiltonian
            # Decide if the weight sample is better than previous. Randomly
            # accept to maintain detailed balance. This decision rule is
            # derived from the Metropolis-Hasting sampling method.
            print ('weights,new_wegihts,M, G, Obj_ERr,Hamil,H', weights,
                new_weights, momentum, new_gradient, new_obj_error, new_H, H
            )
            if (h_diff < 0) or (np.random.uniform() < np.exp(-h_diff)): # metropolis decision rule + random accept
                gradient = new_gradient
                weights = new_weights
                obj_error = new_obj_error
            self.store_weights(weights)
            self.weights = weights
