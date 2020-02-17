# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np

from ift725.layers import *
from ift725.layer_combo import *


class TwoLayerNeuralNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be FC - relu - FC - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input (3x32x32 for a CIFAR10 image)
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialisez les poids et les biais du réseau de neuronnes à deux   #
        #  couches.                                                                #
        # Les poids devraient être initialisés à partir d'une Gaussienne dont      #
        # l'écart-type est égal à weight_scale et les biais devraient être         #
        # initialisés à zéro. Tous les poids et les biais devraient être stockés   #
        # dans le dictionnaire `self.params`. Les poids et les biais de la première#
        # couche devraient utiliser les clés 'W1' et 'b1' respectivement et les    #
        # poids et les biais de la seconde couche devraient utiliser les clés 'W2' #
        # et 'b2'.  En d'autres mots, votre code devrait être comme suit:          #
        # self.params['W1'] = ...                                                  #
        # self.params['b1'] = ...                                                  #
        # ...                                                                      #
        ############################################################################


        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        RECALL : the architectur is  FC - relu - FC - softmax.


        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        N is the size of the batch

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implémentez la propagation avant pour le réseau de neuronnes à deux#
        #  couches, en calculant les scores de classes pour X. Stockez les scores  #
        #  dans la variable `scores`.  N'oubliez pas de conserver les caches des   #
        #  deux couches.                                                           #
        #  NOTES: score est la sortie du réseau *SANS SOFTMAX*                     #
        ############################################################################

        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implémentez la rétropropagation pour ce réseau de neuronnes à deux #
        #  couches.                                                                #
        # Stockez la perte dans la variable loss et les gradients dans le          #
        # dictionnaire grads. Calculez la perte sur les données en utilisant       #
        # softmax et assurez-vous que grads[k] contient les gradients pour         #
        # self.params[k]. N'oubliez pas d'ajouter la régularisation L2!            #
        #                                                                          #
        # NOTE: Afin de vous assurer que votre implémentation équivaut à la        #
        # solution et qu'elle passe les tests automatiques, assurez-vous que votre #
        # régularisation L2 inclue un facteur de 0.5 afin de simplifier            #
        # l'expression pour le gradient.                                           #
        #                                                                          #
        # À faire :                                                                #
        # 1- Calculer la loss du softmax ainsi que son gradient                    #
        # 2- Rétro-progagez le gradient à travers le 2e couche pleinement connectée#
        # 3- Rétro-progagez le gradient à travers le 1ere couche pleinement connectée#
        # 4- Ajoutez la régularisation L2 aux paramètres                           #
        #                                                                          #
        # Note, les gradients doivent être stochez dans le dictionnaire `grads`    #
        #       du type grads['W1']=...                                            #
        ############################################################################
        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

        return loss, grads


class FullyConnectedNeuralNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {FC - [batch norm] - relu - [dropout]} x (L - 1) - FC - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialisez les paramètres du réseau de neuronnes en stockant      #
        #  toutes les valeurs dans le dictionnaire self.params.                    #
        # Stockez les poids et les biais de la première couche dans 'W1' et 'b1'   #
        # respectivement; pour la deuxième couche, utilisez 'W2' et 'b2', etc.     #
        # Les poids devraient être initialisés à partir d'une distribution normale #
        # d'écart-type égal à weight_scale et les biais devraient être initialisés #
        # à zéro.                                                                  #
        #                                                                          #
        # Lorsque batch norm est utilisé, stockez les paramètres de mises à l'échelle  #
        # et les décalages (gamma et beta) pour la première couche dans gamma1     #
        # et beta1; pour la deuxième couche, utilisez gamma2 et beta2, etc. Les    #
        # paramètres gamma devraient être initialisés à 1 et beta à zéro.          #
        #
        # NOTE: utilisez la fonction self.pn pour retrouver les clé du dictionnaire#
        #       params.  Par exemple, pour accéder aux poids de la couche 4, vous  #
        #       pouvez faire :                                                     #
        #           param_name_W = self.pn('W',4)                                  #
        #           param_name_b = self.pn('b',4)                                  #
        #           self.params[param_name_W] = ...                                #
        #           self.params[param_name_b] = ...                                #
        ############################################################################

        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            if type(v) is list:
                self.params[k] = [x.astype(dtype) for x in v]
            else:
                self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implémentez la propagation avant pour le réseau de neurones pleinement  #
        #  connectées. Calculez les scores de classes pour X et stockez les dans   #
        #  la variable `scores`.                                                   #
        #                                                                          #
        # Lors de l'utilisation de dropout, vous devrez passer self.dropout_param  #
        # à chacune des propagation dropout.                                       #
        #                                                                          #
        # Lors de l'utilisation de batch normalization, vous devrez passer         #
        # self.bn_params[0] à chacune des propagation pour la première couche de   #
        # normalisation par lots; passer self.bn_params[1] pour la propagation de  #
        # la deuxième couche de normalisation par lots, etc.                       #
        ############################################################################

        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implémentez la rétropropagation pour le réseau de neuronnes        #
        #  pleinement connectées. Stockez la perte dans la variable loss et les    #
        #  gradients dans le dictionnaire grads. Calculez la perte des données en  #
        #  utilisant softmax et assurez-vous que grads[k] contient les gradients   #
        #  pour self.params[k]. N'oubliez pas la régularisation L2!                #
        #                                                                          #
        # Lors de l'utilisation de la normalisation par lots, vous n'avez pas à    #
        # régulariser les paramètres de mise à l'échelle et de décalage.           #
        #                                                                          #
        # NOTE: Afin de vous assurer que votre implémentation équivaut à la        #
        # solution et que vous passiez les tests automatisé, assurez-vous que la   #
        # régularisation L2 inclus un facteur de 0.5 pour simplifier l'expression  #
        # pour le gradient.                                                        #
        ############################################################################

        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

        return loss, grads

    def pn(self, name, i):
        return name + str(i if i != -1 else self.num_layers)
