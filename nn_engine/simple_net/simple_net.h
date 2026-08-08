#pragma once

#include "simple_layer.h"

struct SimpleNet {
    std::vector<size_t> layerSizes;
    std::vector<SimpleLayer> layers;
    
    SimpleNet() : layerSizes(2, 1)
                , layers(2, SimpleLayer(1, 1))
    {}
};