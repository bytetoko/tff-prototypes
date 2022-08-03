import tensorflow as tf
import tensorflow_federated as tff

@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)

def main():
    # This is a simple snippet showing how we can get the proto representation of
    # a federated computation.
    # As noted in the tutorial (https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1#executing_federated_computations)
    # At definition time, function wrapped in tff.federated_computation is traced
    # and converted into a ConcreteComputation
    # (https://github.com/tensorflow/federated/blob/0829df1107c9918938170904739aa94a1f92b4e0/tensorflow_federated/python/core/impl/computation/computation_impl.py#L32)
    # Which is a wrapper for the Computation proto
    # (https://github.com/tensorflow/federated/blob/0829df1107c9918938170904739aa94a1f92b4e0/tensorflow_federated/python/core/impl/computation/computation_base.py#L21)

    print(f"The class for get_average_temperature is {get_average_temperature.__class__}")

    # get_proto is a class method so we have to pass the instance of the proto as an argument
    proto = get_average_temperature.get_proto(get_average_temperature)
    print(f"The protocol buffer is {proto}")

if __name__ == "__main__":
    main()