from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)

        # assign random weights to my inputs, a 3x1 matrix
        random_weights = random.random((3,1))

        # normalize weights over -1 to 1
        self.synaptic_weights = 2 * random_weights - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self,x):
        return x * (1 - x)

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, training_set_inputs, training_set_outputs, iterations):
        for i in xrange(iterations):
            output = self.think(training_set_inputs)

            # absolute error
            error = training_set_outputs - output

            # product of inputs with absolute error, gradiant descent
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # adjust
            self.synaptic_weights += adjustment

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    print "Considering output of new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))
