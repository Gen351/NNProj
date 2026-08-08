#pragma once

#include<fstream>
#include<memory>
#include<stdexcept>
#include<string>
#include<vector>

#include "../simple_net/simple_net.h"
#include "../simple_net/simple_layer.h"
#include "../simple_net/simple_activation_layer.h"

// Central place for saving/loading a SimpleNet as a human-readable text file.
// Save/load with just a name; the ".simple_net" extension is appended automatically.
//
// File format (all numbers whitespace-separated, newlines irrelevant):
//   SIMPLE_NET <version>
//   <layerCount>
//   LAY <input> <output>
//   <input*output weights, row-major> <output biases>
//   ACT <SIGM|RELU|LEAK|TANH>
//   ...
namespace SimpleNetReader {

namespace {
    constexpr int FORMAT_VERSION = 1;

    inline std::string withExtension(const std::string& name) {
        const std::string ext = ".simple_net";
        if(name.size() >= ext.size()
           && name.compare(name.size() - ext.size(), ext.size(), ext) == 0)
            return name;
        return name + ext;
    }

    inline std::string activationToString(SimpleActivationLayer::ActivationType type) {
        switch(type) {
            case SimpleActivationLayer::ActivationType::SIGM: return "SIGM";
            case SimpleActivationLayer::ActivationType::RELU: return "RELU";
            case SimpleActivationLayer::ActivationType::LEAK: return "LEAK";
            case SimpleActivationLayer::ActivationType::TANH: return "TANH";
        }
        throw std::runtime_error("SimpleNetReader: unknown activation type");
    }

    inline SimpleActivationLayer::ActivationType stringToActivation(const std::string& tag) {
        if(tag == "SIGM") return SimpleActivationLayer::ActivationType::SIGM;
        if(tag == "RELU") return SimpleActivationLayer::ActivationType::RELU;
        if(tag == "LEAK") return SimpleActivationLayer::ActivationType::LEAK;
        if(tag == "TANH") return SimpleActivationLayer::ActivationType::TANH;
        throw std::runtime_error("SimpleNetReader: unknown activation type '" + tag + "'");
    }

    inline std::unique_ptr<AbstractSimpleLayer> loadLayer(std::ifstream& file, size_t index) {
        size_t in = 0, out = 0;
        if(!(file >> in >> out) || in == 0 || out == 0)
            throw std::runtime_error("SimpleNetReader::load: invalid dimensions at layer "
                                     + std::to_string(index));

        auto layer = std::make_unique<SimpleLayer>(in, out);
        for(float& w : layer->weights.data)
            if(!(file >> w))
                throw std::runtime_error("SimpleNetReader::load: missing weight at layer "
                                         + std::to_string(index));
        for(float& b : layer->biases)
            if(!(file >> b))
                throw std::runtime_error("SimpleNetReader::load: missing bias at layer "
                                         + std::to_string(index));
        return layer;
    }

    inline std::unique_ptr<AbstractSimpleLayer> loadActivation(std::ifstream& file, size_t index) {
        std::string tag;
        if(!(file >> tag))
            throw std::runtime_error("SimpleNetReader::load: missing activation type at layer "
                                     + std::to_string(index));

        auto layer = std::make_unique<SimpleActivationLayer>();
        layer->setActivation(stringToActivation(tag));
        return layer;
    }

    // layerSizes holds the boundary sizes: [in0, out0, out1, ...]
    inline void rebuildLayerSizes(SimpleNet& net) {
        net.layerSizes.clear();
        bool haveFirst = false;
        for(const auto& layer : net.layers) {
            if(layer->getType() != "LAY") continue;
            const SimpleLayer& dense = static_cast<const SimpleLayer&>(*layer);
            if(!haveFirst) {
                net.layerSizes.push_back(dense.weights.cols());
                haveFirst = true;
            }
            net.layerSizes.push_back(dense.weights.rows());
        }
    }
}

    /// @brief Writes net to "<name>.simple_net" (extension appended if missing).
    inline void save(const std::string& name, const SimpleNet& net) {
        const std::string path = withExtension(name);
        std::ofstream file(path);
        if(!file.is_open())
            throw std::runtime_error("SimpleNetReader::save: failed to open '" + path + "'");

        file << "SIMPLE_NET " << FORMAT_VERSION << "\n";
        file << net.layers.size() << "\n";

        for(size_t i = 0; i < net.layers.size(); i++) {
            const AbstractSimpleLayer& layer = *net.layers[i];

            if(layer.getType() == "LAY") {
                const SimpleLayer& dense = static_cast<const SimpleLayer&>(layer);
                if(dense.biases.size() != dense.weights.rows())
                    throw std::runtime_error("SimpleNetReader::save: bias/weight mismatch at layer "
                                             + std::to_string(i));

                file << "LAY " << dense.weights.cols() << " " << dense.weights.rows() << "\n";
                for(const float& w : dense.weights.data) file << w << " ";
                for(const float& b : dense.biases) file << b << " ";
                file << "\n";
            }
            else if(layer.getType() == "ACT") {
                const SimpleActivationLayer& act = static_cast<const SimpleActivationLayer&>(layer);
                file << "ACT " << activationToString(act.getActivation()) << "\n";
            }
            else {
                throw std::runtime_error("SimpleNetReader::save: unknown layer type at layer "
                                         + std::to_string(i));
            }
        }

        if(!file)
            throw std::runtime_error("SimpleNetReader::save: failed writing to '" + path + "'");
    }

    /// @brief Reads "<name>.simple_net" (extension appended if missing) into a new SimpleNet.
    inline SimpleNet load(const std::string& name) {
        const std::string path = withExtension(name);
        std::ifstream file(path);
        if(!file.is_open())
            throw std::runtime_error("SimpleNetReader::load: failed to open '" + path + "'");

        std::string magic;
        int version = 0;
        if(!(file >> magic >> version))
            throw std::runtime_error("SimpleNetReader::load: '" + path + "' is not a .simple_net file");
        if(magic != "SIMPLE_NET")
            throw std::runtime_error("SimpleNetReader::load: '" + path + "' is not a .simple_net file");
        if(version != FORMAT_VERSION)
            throw std::runtime_error("SimpleNetReader::load: unsupported version "
                                     + std::to_string(version) + " in '" + path + "'");

        size_t layerCount = 0;
        if(!(file >> layerCount))
            throw std::runtime_error("SimpleNetReader::load: missing layer count in '" + path + "'");

        SimpleNet net;
        net.layers.clear();
        for(size_t i = 0; i < layerCount; i++) {
            std::string tag;
            if(!(file >> tag))
                throw std::runtime_error("SimpleNetReader::load: unexpected end of file at layer "
                                         + std::to_string(i));

            if(tag == "LAY") net.layers.push_back(loadLayer(file, i));
            else if(tag == "ACT") net.layers.push_back(loadActivation(file, i));
            else throw std::runtime_error("SimpleNetReader::load: unknown layer type '" + tag
                                          + "' at layer " + std::to_string(i));
        }

        rebuildLayerSizes(net);
        return net;
    }

}
