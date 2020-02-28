# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np



def forward_fully_connected(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    N: batch size
    d_i : number of elements in dimension i
    D : number of elements in one batch
    M : number of neurons at the output of the layer

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implémentez la propagation avant d'une couche pleinement connectée. #
    #  Stockez le résultat dans out.                                            #
    # Vous devrez reformer les entrées en lignes.                               #
    #############################################################################
    inlineX = np.array([a.flatten() for a in x])
    out = inlineX @ w + b
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def backward_fully_connected(dout, cache):
    """
    Computes the backward pass for a fully connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    N: batch size
    d_i : number of elements in dimension i
    D : number of elements in each batch x_i.  D = d_1*d_2*...*d_k (D=32x32x3 for a CIFAR10 image)
    M : number of neurons at the output of the layer

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implémentez la rétropropagation pour une couche pleinement          #
    #  connectée.                                                               #
    #############################################################################
    inlineX = np.array([a.flatten() for a in x])
    dx = np.transpose(w@dout.T).reshape(x.shape)
    dw = np.transpose(dout.T@inlineX)
    db = np.sum(dout, axis=0, keepdims=False)
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    return dx, dw, db


def forward_relu(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implémentez la propagation pour une couche ReLU.                    #
    #############################################################################
    out = np.array([a.flatten() for a in x])
    out[out<0] = 0
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    cache = x > 0
    return out, cache


def backward_relu(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to the input x
    """
    dx = None
    #############################################################################
    # TODO: Implémentez la rétropropagation pour une couche ReLU.               #
    #############################################################################
    dx=np.multiply(dout,cache)
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    return dx


def forward_batch_normalization(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # Batch normalization en mode TRAIN                                         #
        #############################################################################
        # Compute mean and var
        batch_mean = np.mean(x, axis=0)  # shape (D,)
        batch_var = np.var(x, axis=0)  # shape (D,)
        # Compute output
        delta = x - batch_mean
        inv_var = 1.0 / (batch_var + eps)
        inv_sqrt_var = np.sqrt(inv_var)
        bn = delta * inv_sqrt_var  # batch norm
        out = gamma[np.newaxis, :] * bn + beta[np.newaxis, :]  # scaling
        cache = {
            'batch_mean': batch_mean,
            'batch_var': batch_var,
            'delta': delta,
            'inv_var': inv_var,
            'inv_sqrt_var': inv_sqrt_var,
            'bn': bn,
            'gamma': gamma
        }
        # Update running mean and var
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var

    elif mode == 'test':
        #############################################################################
        # Batch normalization en mode TEST                                          #
        #############################################################################
        out = (x - running_mean) / np.sqrt(running_var + eps)  # batch norm
        out = gamma * out + beta  # scaling

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def backward_batch_normalization(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    gamma = cache['gamma']
    inv_sqrt_var = cache['inv_sqrt_var']

    dout_dxn = dout * gamma  # (N, D)
    dout_dv = np.sum(dout_dxn * cache['delta'], axis=0) * (-1.0 / 2) * inv_sqrt_var ** 3  # (D,)
    dout_ddelta_v = dout_dv * (2.0 / N) * cache['delta']  # (N, D)
    dout_ddelta_d = dout_dxn * inv_sqrt_var  # (N, D)
    dout_dmu_d = -1 * np.sum(dout_ddelta_d, axis=0)  # (D,)
    dout_dmu_v = -1 * np.sum(dout_ddelta_v, axis=0)  # (D,)
    dout_dx_m = (dout_dmu_d + dout_dmu_v) * (1.0 / N)  # (D,)
    dout_dx_0 = dout_ddelta_d + dout_ddelta_v  # (N, D)
    dx = dout_dx_0 + dout_dx_m

    dout_dgamma = cache['bn']
    dgamma = np.sum(dout * dout_dgamma, axis=0)

    dout_dbeta = np.array(1)
    dbeta = np.sum(dout * dout_dbeta, axis=0)

    return dx, dgamma, dbeta


def backward_batch_normalization_alternative(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as backward_batch_normalization, but might not use all of the values in the cache.

    Inputs / outputs: Same as backward_batch_normalization
    """
    dx, dgamma, dbeta = None, None, None

    N, D = dout.shape

    gamma = cache['gamma']
    inv_var = cache['inv_var']
    inv_sqrt_var = cache['inv_sqrt_var']
    delta = cache['delta']
    term1 = N * dout
    term2 = np.sum(dout, axis=0)
    term3 = delta * inv_var * np.sum(dout * delta, axis=0)
    dx = (1.0 / N) * gamma * inv_sqrt_var * (term1 - term2 - term3)

    dout_dgamma = cache['bn']
    dgamma = np.sum(dout * dout_dgamma, axis=0)

    dout_dbeta = np.array(1)
    dbeta = np.sum(dout * dout_dbeta, axis=0)

    return dx, dgamma, dbeta


def forward_inverted_dropout(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p_drop, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None
    p = 1.0 - p_drop

    ###########################################################################
    # TODO: Implémentez la propagation avant pour la phase d'entrainement     #
    #  le dropout inversé (inverted dropout).                                 #
    # Stockez le masque de dropout dans la variable mask.                     #
    # NOTE : le dropout "normal" impose qu'en mode test, la sortie soit       #
    #        multipliée par `p`.  Le dropout inversé divise la sortie par `p` #
    #        en mode 'train'.  https://deepnotes.io/dropout
    ###########################################################################
    if mode == 'train':
        mask = np.random.binomial(1, p, size=x.shape) / p
        out = x * mask

    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)

    ###########################################################################
    #                            FIN DE VOTRE CODE                            #
    ###########################################################################
    return out, cache


def backward_inverted_dropout(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None

    ###########################################################################
    # TODO: Implémentez la rétropropagation pour la phase d'entrainement pour #
    #  le dropout inversé (inverted dropout).                                 #
    ###########################################################################
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout

    ###########################################################################
    #                            FIN DE VOTRE CODE                            #
    ###########################################################################

    return dx


def forward_convolutional_naive(x, w, b, conv_param, verbose=0):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of a batch of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implémentez la propagation pour la couche de convolution.           #
    # Astuces: vous pouvez utiliser la fonction np.pad pour le remplissage.     #
    #############################################################################

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    pad, stride = conv_param['pad'], conv_param['stride']

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_out, W_out))


    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    padded_x = np.pad(x, pad_width, 'constant', constant_values = 0)


    for image in range(N):
        for filtre in range(F):
            for height in range(H_out):
                for width in range(W_out):
                    height_stride = height * stride
                    width_stride = width * stride
                    out[image, filtre, height, width] = np.sum(
                    padded_x[image, :, height_stride: height_stride + HH, width_stride: width_stride + WW] * w[filtre])+ b[filtre]

    cache = (x, w, b, conv_param)


    return out, cache

    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    cache = None

    return out, cache


def backward_convolutional_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.  (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x : (N, C, H, W)
    - dw: Gradient with respect to w : (F, C, HH, WW)
    - db: Gradient with respect to b : (F,)
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implémentez la rétropropagation pour la couche de convolution       #
    #############################################################################

    x, w, b, conv_param = cache
    dw, db = np.zeros_like(w), np.zeros_like(b)

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    N, F, H_out, W_out = dout.shape


    pad, stride = conv_param['pad'], conv_param['stride']
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    padded_x = np.pad(x, pad_width, mode='constant', constant_values=0)

    padded_dx = np.zeros_like(padded_x)


    for image in range(N):
        for filtre in range(F):
            for height in range(H_out):
                for width in range(W_out):
                    height_stride = height * stride
                    width_stride = width * stride
                    db[filtre] += dout[image, filtre, height, width]
                    dw[filtre] += padded_x[image, :,height_stride:height_stride+HH, width_stride:width_stride + WW] * dout[image, filtre, height, width]

                    padded_dx[image, :, height_stride:height_stride + HH, width_stride:width_stride + WW] += w[filtre] * dout[image, filtre, height, width]

    dx = padded_dx[:, :, pad:pad + H, pad:pad + W]

    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    return dx, dw, db


def forward_max_pooling_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    #############################################################################
    # TODO: Implémentez la propagation pour une couche de de max pooling        #
    #############################################################################

    N, C, H, W = x.shape

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = (H - pool_height) // stride + 1
    W_out = (W - pool_width) // stride + 1

    out = np.zeros((N, C, H_out, W_out))

    # Pooling
    for image in range(N):
        for c in range(C):
            for height in range(H_out):
                for width in range(W_out):
                    height_stride = height * stride
                    width_stride = width * stride
                    out[image, c, height, width] = np.max(
                        x[image, c, height_stride:height_stride + pool_height, width_stride: width_stride + pool_width])

    cache = (x, pool_param)

    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################

    return out, cache


def backward_max_pooling_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implémentez la rétropropagation pour une couche de max pooling.     #
    #############################################################################

    x, pool_param = cache
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = (H - HH) // stride + 1
    W_out = (W - WW) // stride + 1
    dx = np.zeros_like(x)

    for image in range(N):
        for c in range(C):
            for height in range(H_out):
                for width in range(W_out):
                    height_stride = height * stride
                    width_stride = width * stride
                    window = x[image, c, height_stride: height_stride + HH, width_stride: width_stride + WW]
                    mask = np.max(window) == window
                    dx[image, c, height_stride: height_stride + HH, width_stride: width_stride + WW] = dout[
                                         image, c, height, width] * mask
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    return dx


def forward_spatial_batch_normalization(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    N, C, H, W = x.shape
    x_2d = np.moveaxis(x, 1, -1)
    x_2d = np.reshape(x_2d, (N * H * W, C))
    out_2d, cache = forward_batch_normalization(x_2d, gamma, beta, bn_param)
    out = np.reshape(out_2d, (N, H, W, C))
    out = np.moveaxis(out, -1, 1)  # --> (N, C, H, W)

    return out, cache


def backward_spatial_batch_normalization(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None


    N, C, H, W = dout.shape
    dout_2d = np.moveaxis(dout, 1, -1)
    dout_2d = np.reshape(dout_2d, (N * H * W, C))
    dx_2d, dgamma, dbeta = backward_batch_normalization(dout_2d, cache) # ou backward_batch_normalization_alternative(dout_2d, cache)
    dx = np.reshape(dx_2d, (N, H, W, C))
    dx = np.moveaxis(dx, -1, 1)  # --> (N, C, H, W)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    ALSO CALLED THE "Hinge Loss"

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    #############################################################################
    # TODO: La perte SVM (ou Hinge Loss) en vous inspirant du tp1 mais sans     #
    #       régularisation                                                      #
    #############################################################################

    hinge = np.maximum(0, 1 + x.T - x[np.arange(N), y]).T
    hinge[np.arange(len(y)), y] = 0
    loss = np.sum(hinge)
    loss /= N
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    margins = np.maximum(0, x - x[np.arange(N), y][:, np.newaxis] + 1.0)
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y, scale=1.0):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N=len(y)
    #############################################################################
    # TODO: La perte softmax en vous inspirant du tp1 mais sans régularisation  #
    #                                                                           #
    #############################################################################
    probs = np.exp(x)
    probs = probs/np.sum(probs, axis=1, keepdims=True)
    loss = np.sum(-np.log(probs[np.arange(len(y)),y]))/len(y)

    #dx = probs@x/len(x)

    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    loss *= scale
    dx *= scale  # useful for gradient checking
    return loss, dx
