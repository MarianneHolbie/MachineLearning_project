# Deep Convolutional Architectures

## TASKS
| Task | Description |
| --- | --- |
| [0. Inception Block](0-inception_block.py) | Write a function `def inception_block(A_prev, filters):` that builds an inception block as described in `Going Deeper with Convolutions (2014)`. |
| [1. Inception Network](1-inception_network.py) | Write a function `def inception_network():` that builds the inception network as described in `Going Deeper with Convolutions (2014)`. |
| [2. Identity Block](2-identity_block.py) | Write a function `def identity_block(A_prev, filters):` that builds an identity block as described in `Deep Residual Learning for Image Recognition (2015)`. |
| [3. Projection Block](3-projection_block.py) | Write a function `def projection_block(A_prev, filters, s=2):` that builds a projection block as described in `Deep Residual Learning for Image Recognition (2015)`. |
| [4. ResNet-50](4-resnet50.py) | Write a function `def resnet50():` that builds the ResNet-50 architecture as described in `Deep Residual Learning for Image Recognition (2015)`. |
| [5. Dense Block](5-dense_block.py) | Write a function `def dense_block(X, nb_filters, growth_rate, layers):` that builds a dense block as described in `Densely Connected Convolutional Networks`. |
| [6. Transition Layer](6-transition_layer.py) | Write a function `def transition_layer(X, nb_filters, compression):` that builds a transition layer as described in `Densely Connected Convolutional Networks`. |
| [7. DenseNet-121](7-densenet121.py) | Write a function `def densenet121(growth_rate=32, compression=1.0):` that builds the DenseNet-121 architecture as described in `Densely Connected Convolutional Networks`. |
