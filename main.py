# Network definition for ResNet-18 on ImageNet dataset
network = [
    {'type': 'conv', 'channels': 64, 'k_size': 7,
        'stride': 2, 'padding': 3, 'activation': 'relu'},
    {'type': 'pool', 'k_size': 3, 'stride': 2, 'padding': 1},
    {'type': 'conv', 'channels': 64, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'conv', 'channels': 64, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'pool', 'k_size': 3, 'stride': 2, 'padding': 1},
    {'type': 'conv', 'channels': 128, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'conv', 'channels': 128, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'pool', 'k_size': 3, 'stride': 2, 'padding': 1},
    {'type': 'conv', 'channels': 256, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'conv', 'channels': 256, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'conv', 'channels': 256, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'pool', 'k_size': 3, 'stride': 2, 'padding': 1},
    {'type': 'conv', 'channels': 512, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'conv', 'channels': 512, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'conv', 'channels': 512, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'pool', 'k_size': 3, 'stride': 2, 'padding': 1},
    {'type': 'conv', 'channels': 512, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'conv', 'channels': 512, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'conv', 'channels': 512, 'k_size': 3,
        'stride': 1, 'padding': 1, 'activation': 'relu'},
    {'type': 'pool', 'k_size': 3, 'stride': 2, 'padding': 1},
    {'type': 'fc', 'size': 4096, 'activation': 'relu'},
    {'type': 'fc', 'size': 4096, 'activation': 'relu'},
    {'type': 'fc', 'size': 1000, 'activation': 'softmax'}
]


def calculate_flop(network):
    flop = 0
    for layer in network:
        if layer['type'] == 'conv':
            flop += layer['channels'] * layer['k_size'] * layer['k_size'] * \
                layer['k_size'] * layer['k_size'] * layer['k_size']
        elif layer['type'] == 'fc':
            flop += layer['size'] * layer['size']
    return flop


print(calculate_flop(network))
