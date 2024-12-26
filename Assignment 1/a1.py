import numpy as np
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout

# Task 1
def light_pixels(image, lightness, channel):
    """Return a mask for each channel that identifies the pixels whose intensity is 
    above the given threshold. The mask of channel i is an array that has values 0 or 1.
    Assume that the image is stored as a numpy array with three channels for the colours
    'red', 'green', and 'blue'.

    Parameters:
    image (numpy.ndarray): Input image array with shape (height, width, 3).
    lightness (int): The intensity threshold.
    channel (str): The channel name ('red', 'green', 'blue').

    Returns:
    numpy.ndarray: Mask array with the same height and width as the input image, 
                   with values 0 or 1 indicating whether the pixel intensity is above the 
                   threshold.

    Examples:
    >>> image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]],
    ...                   [[  2,  20,  20], [250, 255, 255], [127, 127, 127]]])                          
    >>> light_pixels(image, 20, 'red')
    array([[1, 0, 0],
           [0, 1, 1]])
    >>> light_pixels(image, 20, 'green')
    array([[0, 1, 0],
           [0, 1, 1]])
    >>> light_pixels(image, 15, 'blue')
    array([[0, 0, 1],
           [1, 1, 1]])
    """
    result = []
    c = {'red':0, 'green':1, 'blue':2}
    idx = c[channel]
    
    for r in image:
        row = []
        for i in range(image.shape[1]):
            if r[i][idx] > lightness:
                row.append(1)
            else:
                row.append(0)
        result.append(row)
    return np.array(result)

# Task 2
def histogram(image, buckets, channel):
    """
    Return a histogram of the channel, where the image is represented as a 
    3-channel numpy array with values between 0 and 255. A histogram is
    an array of length `buckets` where the i-th element is the count of pixels
    in the range [i * (256 // buckets), (i + 1) * (256 // buckets)).

    This function should not use third-party functions such as np.linspace or
    np.histogram.

    Parameters:
    image (numpy.ndarray): Input image array with shape (height, width, 3).
    buckets (int): Number of histogram buckets.
    channel (str): The channel name ('red', 'green', 'blue').

    Returns:
    numpy.ndarray: Histogram array of length `buckets`.

    Examples:
    >>> image = np.array([[[250,   2,   2], [  0,   2, 255], [  0,   0, 255]], 
    ...                   [[  2,   2,  20], [250, 255, 255], [127, 127, 127]]])
    >>> histogram(image, 4, 'red')
    array([3, 1, 0, 2])
    >>> histogram(image, 5, 'green')
    array([4, 0, 1, 0, 1])
    >>> histogram(image, 6, 'blue')
    array([2, 0, 0, 1, 0, 3])
    """
    temp = []
    result = []
    c = {'red':0, 'green':1, 'blue':2}
    idx = c[channel]
    
    for r in image:
        for i in range(image.shape[1]):
            temp.append(r[i][idx])
            
    for i in range(buckets):
        count = 0
        for j in range(len(temp)):
            if (i == buckets-1):
                count = len(temp)-sum(result)
            elif (temp[j] >= (i * (256 // buckets))) & (temp[j] < ((i + 1) * (256 // buckets))):
                count += 1
        result.append(count)
    return np.array(result)
     
# Task 3
def build_deep_nn(rows, columns, channels, layer_options):
       #num_hidden, hidden_sizes, dropout_rates,
       #           output_size, output_activation):
    """Return a Keras neural model that has the following layers:
    - a Flatten layer with input shape (rows, columns, channels)
    - as many hidden layers as the length of layer_options
    - layer_options is a list of layer options, such that:
      - hidden layer number i is of size layer_options[i][0] and activation
        layer_options[i][1]
      - if layer_options[i][2] > 0, then hidden layer number i is followed
        `by a dropout layer with dropout rate layer_options[i][2]

    Parameters:
    rows (int): Number of rows in the input.
    columns (int): Number of columns in the input.
    channels (int): Number of channels in the input.
    layer_options (list of tuples): Each tuple contains three elements:
                                   (hidden_size, activation, dropout_rate).

    Returns:
    keras.models.Sequential: The constructed Keras model.
     
    Examples:
    >>> rows = 28
    >>> columns = 28
    >>> channels = 2
    >>> layer_options = [
    ...     (128, 'relu', 0.2),  # 128 neurons, relu activation, 20% dropout
    ...     (64, 'relu', 0),     # 64 neurons, relu activation, no dropout
    ...     (32, 'sigmoid', 0.4) # 32 neurons, sigmoid activation, 40% dropout
    ... ]
    >>> model = build_deep_nn(rows, columns, channels, layer_options)
    >>> model.summary()
    Model: "sequential"
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
    │ flatten (Flatten)                    │ (None, 1568)                │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense (Dense)                        │ (None, 128)                 │         200,832 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dropout (Dropout)                    │ (None, 128)                 │               0 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense_1 (Dense)                      │ (None, 64)                  │           8,256 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dense_2 (Dense)                      │ (None, 32)                  │           2,080 │
    ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
    │ dropout_1 (Dropout)                  │ (None, 32)                  │               0 │
    └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
    Total params: 211,168 (824.88 KB)
    Trainable params: 211,168 (824.88 KB)
    Non-trainable params: 0 (0.00 B)
    >>> model.layers[1].get_config()['activation']
    'relu'
    >>> model.layers[3].get_config()['activation']
    'relu'
    >>> model.layers[4].get_config()['activation']
    'sigmoid'
    """
    network = Sequential()
    network.add(Input(shape=(rows, columns, channels)))
    network.add(Flatten())
    for i in range(len(layer_options)):
        network.add(Dense(layer_options[i][0], activation=layer_options[i][1]))
        if layer_options[i][2] > 0:
            network.add(Dropout(layer_options[i][2]))
    return network

if __name__ == "__main__":
    import doctest
    doctest.testmod()
