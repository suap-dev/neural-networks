from warnings import catch_warnings
import numba as nba # to compute quickly
import numpy as np  # to compute every numerical operation


class Network:
    def __init__(
            self,
            single_stimulus_size=1,
            nr_values_to_predict=1,
            hidden_layer_size=6,
            hidden_layers=1
        ):        
        self.stimulus_size = single_stimulus_size
        self.hidden_neurons = hidden_layer_size
        self.expected_predictions = nr_values_to_predict

        self.weights_0 = np.random.random(size=(
                    self.stimulus_size,
                    self.hidden_neurons
                ))
        self.biases_0 = np.random.random(size=self.hidden_neurons)

        self.weights_1 = np.random.random(size=(
                    self.hidden_neurons,
                    self.expected_predictions
                ))
        self.biases_1 = np.random.random(size=self.expected_predictions)

        self.stimulus_container = np.zeros(single_stimulus_size)


    @nba.vectorize(nopython=True)
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @nba.vectorize(nopython=True)
    def relu(x):
        return x if x >= 0 else 0

    # @nba.njit
    def predict(self, input):
        try:
            assert self.stimulus_size == len(input), "The size of the sample you provided differs from one this network can process."

            self.stimulus_container[:] = input
            return self.think(input)
        except AssertionError as IncorrectDataSize:
            raise IncorrectDataSize

    def process(self, input, weights, biases):
        return self.sigmoid(np.dot(input, weights) + biases)

    # @nba.njit
    def think(self, input):
        return self.process(
                self.process(
                    input,
                    self.weights_0,
                    self.biases_0
                ),
                self.weights_1,
                self.biases_1
            )
    
    # @nba.njit
    def cost(self, predictions, expected_values):
        sum_cost = 0.0
        for i in np.arange(self.expected_predictions):
            sum_cost += (predictions[i]-expected_values[i])**2
        return sum_cost


    def single_experience(self, input, expected_values):
        try:
            assert self.stimulus_size == len(input), "The size of the sample you provided differs from one this network can process."
            assert self.expected_predictions == len(expected_values), f"This network can predict {self.expected_predictions} values, and you provided {expected_values} that you expect."

            #TODO: Implement cost funcsion            
            return self.cost(self.think(input), expected_values)
        except AssertionError as IncorrectDataSize:
            raise IncorrectDataSize

if __name__ == "__main__":
    nn = Network(3, 3)  # Network(single input/stimulus size, number of expected predictions)
    predictions = nn.predict([3.4, 1, 1.2])
    nn.single_experience([0,0,1], [0,0,1])
    print(predictions)