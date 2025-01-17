'''
Builds a model with multiple hidden fully-connected layers.
    model_settings (dict): Dictionary containing different settings for model training.
        - 'fingerprint_size' (int): Size of the input features.
        - 'label_count' (int): Number of output labels.
    model_size_info (list): List where the length defines the number of hidden layers and
        each element represents the number of neurons in that layer.
    tf.keras.Model: A Keras Model instance of the 'DNN' architecture.
'''
import tensorflow as tf

def create_dnn_model(model_settings, model_size_info):
    """Builds a model with multiple hidden fully-connected layers.

    For details see https://arxiv.org/abs/1711.07128.

    Args:
        model_settings: Dict of different settings for model training.
        model_size_info: Length of the array defines the number of hidden-layers and
            each element in the array represent the number of neurons in that layer.

    Returns:
        tf.keras Model of the 'DNN' architecture.
    """

    inputs = tf.keras.Input(shape=(model_settings['fingerprint_size'], ), name='input')

    # First fully connected layer.
    x = tf.keras.layers.Dense(units=model_size_info[0], activation='relu')(inputs)

    # Hidden layers with ReLU activations.
    for i in range(1, len(model_size_info)):
        x = tf.keras.layers.Dense(units=model_size_info[i], activation='relu')(x)

    # Output fully connected layer.
    output = tf.keras.layers.Dense(units=model_settings['label_count'], activation='softmax')(x)

    return tf.keras.Model(inputs, output)
