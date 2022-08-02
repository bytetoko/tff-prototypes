"""MNIST tutorial

This is the first half of the mnist tutorial
https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification

But structured as a python module as opposed to Jupyter notebook.

The first half of the tutorial uses Keras Models and then tff.learning.from_keras_model
to construct the TFF model computation.

The second half (https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#customizing_the_model_implementation)
which is not in this script uses the lower level tff.learning.Model

"""
import tensorflow as tf
import tensorflow_federated as tff
import collections

NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def make_federated_data(client_data, client_ids):
    """Create a list of TF.Datasets
    
    Each element in the list holds the data of a different client to be fed to the
    simulation.

    """
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]

def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])

def create_model_fun(input_spec):
    """Create a model function.
    
    Args:
      input_spec: The spec for the input
    """
    def model_fn():
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        keras_model = create_keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model_fn

def main():
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

    federated_train_data = make_federated_data(emnist_train, sample_clients)

    print(f'Number of client datasets: {len(federated_train_data)}')
    print(f'First dataset: {federated_train_data[0]}')

    # Create an example dataset based on client ID 0
    # This will be used to get the inputspec.
    # N.B. I think what we are doing hear is pushing a sample of data through the preprocessing pipeline
    # and then using that example output to get its spec to pass to the model.
    example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
    preprocessed_example_dataset = preprocess(example_dataset)
    input_spec = preprocessed_example_dataset.element_spec

    model_fn = create_model_fun(input_spec)

    # Construct a federated computation
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    # Initialize constructs the server state (e.g. initializes global model parameters)
    state = iterative_process.initialize()

    # Run  federated training    
    NUM_ROUNDS = 11
    for round_num in range(1, NUM_ROUNDS):
        # Next invokes one round. I think it takes care of distributing current
        # state (model parameters) to each worker, running the computations using
        # the local data and then aggregating the results to produce the new state
        result = iterative_process.next(state, federated_train_data)
        state = result.state
        metrics = result.metrics
        print('round {:2d}, metrics={}'.format(round_num, metrics))

if __name__ == "__main__":
    main()