class HMCNeuron(object):
    """

    The classification problem is where you have some input data, and you want
    to find some functional relationship between this data and it's underlying
    classification.

    A Hamiltonian Monte Carlo (HMC) binary classifier is an approach to making
    bayesian predictions for the classification problem. By using a gradient
    descent algorithm with added noise, the HMC neuron is able to find samples
    of weights, from their empirical distribution, that show the relationship
    between input data and their target classes. By integrating over these
    samples (and marginalizing over all other parameters), predictions and
    your belief of those predictions can be made.

    [HMC vs Naive gradient descent (ie: steepest descent)]
    A steepest descent algorithm is a function that iteratively minimizes some
    objective function. The objective function in this case is the predicted
    errors. Steepest descent works by calculating the gradient of this
    objective function, and updating the weights in the direction of the
    gradient. Because only the gradient is used in updating the weights
    (in steepest descent), the weights will only move in the direction where
    the objective function (the error) is minimized. This causes problems like:

    1. The weights will converge to the most probable, error minimized point.
    2. The weights can grow larger or smaller forever creating strict decision
       boundaries.

    Both of these problems lead to over-fitting, which is making over-confident
    predictions. A model that over-fits, does not generalize well to new data.
    Also, if we just use the most probable weights for our predictions,
    we leave out information about the distribution of the weights. The most
    probable weights are not representative of the typical sample of weights.
    Wouldn't you want to know what portion of the weight distribution classifies
    a point one way or the other? Visually, steepest descent has straight-line
    contours between classifications boundaries. In reality, there are a
    distribution of weights and the contours are less rigid (especially where
    there is a low input sample). Hamiltonian Monte Carlo is able to explore
    this weight distribution and get samples from it by employing a random walk
    strategy and making many gradient calculations each step. A step is simply
    accepting the updated weights. By adding a random walk to your gradient
    descent, you are able to explore more than just the most probable weights.
    Finally, by accepting some steps with a random probability, you can ensure
    that detailed balance holds.
    """
