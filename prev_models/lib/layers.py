from builtins import range
import numpy as np

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0,x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = dout
    dx[x<=0] = 0
    return dx


def conv_forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, H, W, C)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = (H + 2 * pad - HH) / stride + 1
      W' = (W + 2 * pad - WW) / stride + 1
    - cache: (x, w, b, conv_param)
    """
    out = None
    # collecting shapes of data/weight
    N, H, W, C = x.shape
    F, C, HH, WW = w.shape

    pad = conv_param['pad']
    stride = conv_param['stride']

    # applying padded formula to H and W
    H_prime = int(1 + (H + 2 * pad - HH) / stride)
    W_prime = int(1 + (W + 2 * pad - WW) / stride)


    # initializes the output to be an array of 0s
    out = np.zeros((N, F, int(H_prime), int(W_prime)))

    # adds the specific padding to the input array (x)
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    padded_x = np.pad(x, pad_width, mode='constant')

    # loops through each image in the batch
    for n in range(N):  
        # applies each filter to the given image in the batch
        for f in range(F):
            # moves the filter down (until it reaches the bottom)
            for i in range(H_prime):
                # moves the filter to the right (until it reaches the right edge)
                for j in range(W_prime):
                    
                    # finds the area we want to apply the convolution operation on
                    row_start = i * stride                # top
                    row_end = row_start + HH              # bottom
                    col_start = j * stride                # left edge
                    col_end = col_start + WW              # right edge
                    x_region = padded_x[n, row_start:row_end, col_start:col_end, :]

                    # Compute convolution operation
                    out[n, f, i, j] = np.sum(x_region * w[f]) + b[f]
    
    x = padded_x # storing x_with_pad instead of just x to fix issues with shape in conv_backward

    cache = (x, w, b, conv_param)
    return out, cache

def max_pool_forward(x, pool_param):
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

    N, H, W, C = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = ((H - pool_height) // stride) + 1
    W_prime = ((W - pool_width) // stride) + 1
    
    # initializes the output to be an array of 0s
    out = np.zeros((N, H_prime, W_prime, C))

    # loops through each image in the batch
    for n in range(N):  
        # applies each filter to the given image in the batch
        for c in range(C):
            # moves the filter down (until it reaches the bottom)
            for i in range(H_prime):
                # moves the filter to the right (until it reaches the right edge)
                for j in range(W_prime):
                    
                    # finds the area we want to apply the convolution operation on
                    row_start = i * stride                # top
                    row_end = row_start + pool_height     # bottom
                    col_start = j * stride                # left edge
                    col_end = col_start + pool_width      # right edge
                    x_region = x[n, row_start:row_end, col_start:col_end, c]

                    # Compute max pooling operation (taking max value from x_region)
                    out[n, c, i, j] = np.max(x_region)

    cache = (x, pool_param)
    return out, cache


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -np.sum(np.log(probs[np.arange(N),y])) / N
    dx = probs
    dx[np.arange(N),y] -= 1
    dx /= N
    return loss, dx