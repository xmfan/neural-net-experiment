from numpy import exp, array, random, dot

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        random_weights = random.random((number_of_inputs_per_neuron, number_of_neurons))
        self.synaptic_weights = 2 * random_weights - 1

class NeuralNetwork():

    # Use an invisible layer to learn combinations of A, B, C
    def __init__(self, layer1, layer2):
        self.layer1, self.layer2 = layer1, layer2

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        layer1_output = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        layer2_output = self.__sigmoid(dot(layer1_output, self.layer2.synaptic_weights))
        return layer1_output, layer2_output

    def train(self, training_set_inputs, training_set_outputs, iterations):
        for _ in xrange(iterations):
            layer1_output, layer2_output = self.think(training_set_inputs)

            layer2_error = training_set_outputs - layer2_output
            layer2_delta = layer2_error * self.__sigmoid_derivative(layer2_output)

            layer1_error = dot(layer2_delta, self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(layer1_output)

            layer1_adjustment = dot(training_set_inputs.T, layer1_delta)
            layer2_adjustment = dot(layer1_output.T, layer2_delta)

            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    def print_weights(self):
        print "Layer 1 (4 neurons, each with 3 inputs): "
        print self.layer1.synaptic_weights
        print "Layer 2 (1 neuron, with 4 inputs):"
        print self.layer2.synaptic_weights


if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(4, 3)

    # Create output layer (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print "Stage 1) Random starting synaptic weights: "
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print "Stage 2) New synaptic weights after training: "
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print "Stage 3) Considering a new situation [1, 1, 0] -> ?: "
    hidden_state, output = neural_network.think(array([1, 1, 0]))

    print output
