import numpy as np

class Value:
    """ encapsulation for a single scalar value and it utilities like gradient"""
    def __init__(self, data, _children=(), _op=''):
        self.data = data 
        self.grad = 0
        # following internal variables are used to create a computation graph
        self._backward = lambda: None   # each value will have a backward function that computes the gradient
        self._prev = set(_children)     #  we use a set to avoid duplicates 
        self._op = _op                  # the operation that prouced this node, for debugging
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # we first define the output for this operation
        output = Value(self.data + other.data, (self, other), '+')
        
        # Then we will define how to compute the gradient for this operation
        def _backward():
            """ we know that add just routes the gradient to both inputs"""
            self.grad += output.grad
            other.grad += output.grad
        # now we wiil attach the backward function to the output
        output._backward = _backward
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        return output
    
    def __pow__(self, other):
        ''' unary operation, only implemented for scalar'''
        assert isinstance(other, (int, float)), "power only implemented for scalar"
        output = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            #                  local gradient               * gradient from above(global)
            self.grad += (other * self.data ** (other - 1)) * output.grad
        output._backward = _backward
        return output
    
    def relu(self):
        output = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (1 if self.data > 0 else 0) * output.grad
        output._backward = _backward
        return output
            
    def tanh(self):
        output = Value(np.tanh(self.data), (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - output ** 2) * output.grad
        output._backward = _backward
        return output
    
    def backward(self):
        ''' we need to sort the nodes in topological order before running the backward pass
        so that we can compute the gradient in the correct order'''
        topological_order = []
        visited = set()
        def construct_topological_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    construct_topological_order(child)
                topological_order.append(v)
        construct_topological_order(self)
        
        # now we can run the backward pass
        self.grad = 1 # the gradient of the output node is 1 of course :)
        for node in reversed(topological_order):
            # the reason we do reverse is that we want to compute the gradient from the output node to the input node
            node._backward()
            
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        ''' this is called when the left operand does not support the operation
        For example, 1 + Value(2), when the object is on the right side of the operator'''
        return self + other
    
    def __sub__(self, other):
        # we dont need to check if other is a Value object, because __add__ will take care of that
        # subtraction is just addition with negative number
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        ''' AS we all know that a/b = a * b**-1 or a * 1/b'''
        return self * other ** -1
    
    def __rtruediv__(self, other):
        return other * self ** -1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    