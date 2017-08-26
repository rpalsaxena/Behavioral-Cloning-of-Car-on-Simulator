# Behavioral-Cloning-of-Car-on-Simulator

| Layer         | Description                                                                               |
|---------------|-------------------------------------------------------------------------------------------|
| Input         | 160x320x3 RGB image                                                                       |
| Cropping2D    | Cropping layer to crop the images                                                         |
| Lambda Layer  | Normalization of pixel values using Lambda layer                                          |
| Convolution2D | 24 filters of 5x 5 dimension, 2x2 stride values with activation function as ReLu          |
| Convolution2D | 36 filters of 5x 5 dimension, 2x2 stride values with activation function as ReLu          |
| Convolution2D | 48 filters of 5x 5 dimension, 2x2 stride values with activation function as ReLu          |
| Convolution2D | 64 filters of 3x3 dimension with default stride val and ReLu activation                   |
| Convolution2D | 88 filters of 3x3 dimension with default stride val and ReLu activation                   |
| MaxPooling2D  | MaxPooling layer to reduce the dimension i.e, over-fitting and increase depth of Network  |
| Flatten       | The final outputs from the above mentioned network is flattened to make a 1D matrix.      |
| Dense         | Fully connected layer 320 output nodes                                                    |
| Dropout       | This helps to reduce probability of overfitting. Reduced 50% of nodes.                    |
| Dense         | Fully connected layer with 100 output nodes                                               |
| Dense         | Fully connected layer with 50 output nodes                                                |
| Dense         | Fully connected layer with 1 output node i.e, final steering angle. :smile:               |
