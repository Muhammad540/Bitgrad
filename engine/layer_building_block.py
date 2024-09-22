import random 
from basic_building_block import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    
class SingleNeuron(Module):
    ''' A single neuraon cares only about the number of inputs it has, the weights and the bias.'''
    def __init__(self, number_input, nonlinearity=True, which_nonlinearity='relu'):
        self.weights =  [Value(random.uniform(-1,1)) for _ in range(number_input)]
        self.bias = Value(0) # as Andrej says this is the measure of happiness of the neuron
        self.non_linearity = nonlinearity
        self.which_nonlinearity = which_nonlinearity
    
    def __call__(self, *args):
        result = sum((weight_i*input_i for weight_i, input_i in zip(self.weights , args)), self.bias) 
        if self.non_linearity:
            if self.which_nonlinearity == 'relu':
                return result.relu()
            elif self.which_nonlinearity == 'tanh':
                return result.tanh()
        else:
            return result
    
    def parameters(self):
        '''return the parameters of the neuron'''
        return self.weights + [self.bias]
    
    def __repr__(self):
        if self.non_linearity:
            return f"{self.which_nonlinearity} neuron with ({len(self.weights)} weights)"
        else:
            return f"Linear neuron with ({len(self.weights)} weights)"
        
def SingleLayer(Module):
    
    def __init__(self, number_of_input_neurons, number_of_output_neurons, nonlinearity=True, which_nonlinearity='relu', **kwargs):
        self.neurons = [SingleNeuron(number_of_input_neurons, which_nonlinearity=which_nonlinearity) for _ in range(number_of_output_neurons)]
    
    def __call__ (self, input):
        layer_output = [neuron(input) for neuron in self.neurons]
        if len(layer_output) == 1:
            return layer_output[0]
        else:
            return layer_output
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron,parameters()]
    
    def __repr__(self):
        return f"layer of [{', '.join(str(neuron) for neuron in self.neurons)}]"

def MultiLayerPerceptron(Module):
    '''
    if you create MultiLayerPerceptron(3, [4, 4, 1]), you'll get a network with:
    3 input neurons
    A hidden layer with 4 neurons
    Another hidden layer with 4 neurons
    An output layer with 1 neuron 
    '''
    def __init__(self, number_of_inputs, number_of_outputs):
        size = [number_of_inputs] + number_of_outputs
        self.layers = [SingleLayer(size[i], size[i+1],nonlinearity=i!=len(number_of_outputs)-1) for i in range(len(number_of_outputs))]
        
    def __call__(self, input):
        ''' this is a forward pass'''
        for layer in self.layers:
            input = layer(input)
        return input
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"