#pragma once

#include<fstream>
#include<memory>
#include<stdexcept>
#include<string>

#include "backprop_net.h"

// Central place for saving/loading a BackpropNet::Net as a human-readable text
// file. Save/load with just a name; the ".backprop_net" extension is appended
// automatically.
//
// File format (all numbers whitespace-separated, newlines irrelevant):
//   BACKPROP_NET <version>
//   <layerCount>
//   DEN <input> <output>
//   <input*output weights, row-major> <output biases>
//   ACT <SIGM|RELU|LEAK|TANH|SOFT>
//   ...
namespace BackpropNetReader {

namespace {
    constexpr int FORMAT_VERSION = 1;
    const std::string EXTENSION = ".backprop_net";

    inline std::string withExtension(const std::string& name) {
        if(name.size() >= EXTENSION.size()
           && name.compare(name.size() - EXTENSION.size(), EXTENSION.size(), EXTENSION) == 0)
            return name;
        return name + EXTENSION;
    }

    inline std::string activationToString(BackpropNet::ActivationLayer::ActType type) {
        switch(type) {
            case BackpropNet::ActivationLayer::ActType::SIGM: return "SIGM";
            case BackpropNet::ActivationLayer::ActType::RELU: return "RELU";
            case BackpropNet::ActivationLayer::ActType::LEAK: return "LEAK";
            case BackpropNet::ActivationLayer::ActType::TANH: return "TANH";
            case BackpropNet::ActivationLayer::ActType::SOFT: return "SOFT";
        }
        throw std::runtime_error("BackpropNetReader: unknown activation type");
    }

    inline BackpropNet::ActivationLayer::ActType stringToActivation(const std::string& tag) {
        if(tag == "SIGM") return BackpropNet::ActivationLayer::ActType::SIGM;
        if(tag == "RELU") return BackpropNet::ActivationLayer::ActType::RELU;
        if(tag == "LEAK") return BackpropNet::ActivationLayer::ActType::LEAK;
        if(tag == "TANH") return BackpropNet::ActivationLayer::ActType::TANH;
        if(tag == "SOFT") return BackpropNet::ActivationLayer::ActType::SOFT;
        throw std::runtime_error("BackpropNetReader: unknown activation type '" + tag + "'");
    }

    inline std::unique_ptr<BackpropNet::AbstractLayer> loadDense(std::ifstream& file, size_t index) {
        size_t in = 0, out = 0;
        if(!(file >> in >> out) || in == 0 || out == 0)
            throw std::runtime_error("BackpropNetReader::load: invalid dimensions at layer "
                                     + std::to_string(index));

        auto layer = std::make_unique<BackpropNet::DenseLayer>(in, out);
        for(float& w : layer->get_weights().data)
            if(!(file >> w))
                throw std::runtime_error("BackpropNetReader::load: file ended prematurely (truncated?) "
                                         "while reading layer " + std::to_string(index));
        for(float& b : layer->get_biases())
            if(!(file >> b))
                throw std::runtime_error("BackpropNetReader::load: file ended prematurely (truncated?) "
                                         "while reading layer " + std::to_string(index));
        return layer;
    }

    inline std::unique_ptr<BackpropNet::AbstractLayer> loadActivation(std::ifstream& file, size_t index) {
        std::string tag;
        if(!(file >> tag))
            throw std::runtime_error("BackpropNetReader::load: missing activation type at layer "
                                     + std::to_string(index));

        return std::make_unique<BackpropNet::ActivationLayer>(stringToActivation(tag));
    }
}

    /// @brief Writes net to "<name>.backprop_net" (extension appended if missing).
    inline void save(const std::string& name, const BackpropNet::Net& net) {
        const std::string path = withExtension(name);
        std::ofstream file(path);
        if(!file.is_open())
            throw std::runtime_error("BackpropNetReader::save: failed to open '" + path + "'");

        file << "BACKPROP_NET " << FORMAT_VERSION << "\n";
        file << net.layers.size() << "\n";

        for(size_t i = 0; i < net.layers.size(); i++) {
            const BackpropNet::AbstractLayer& layer = *net.layers[i];
            const std::string type = layer.get_type();

            if(type == "DEN") {
                const auto& dense = static_cast<const BackpropNet::DenseLayer&>(layer);
                if(dense.get_biases().size() != dense.get_weights().rows())
                    throw std::runtime_error("BackpropNetReader::save: bias/weight mismatch at layer "
                                             + std::to_string(i));

                file << "DEN " << dense.get_weights().cols() << " " << dense.get_weights().rows() << "\n";
                for(const float& w : dense.get_weights().data) file << w << " ";
                for(const float& b : dense.get_biases()) file << b << " ";
                file << "\n";
            }
            else if(type == "ACT") {
                const auto& act = static_cast<const BackpropNet::ActivationLayer&>(layer);
                file << "ACT " << activationToString(act.get_activation()) << "\n";
            }
            else {
                throw std::runtime_error("BackpropNetReader::save: unknown layer type at layer "
                                         + std::to_string(i));
            }
        }

        if(!file)
            throw std::runtime_error("BackpropNetReader::save: failed writing to '" + path + "'");
    }

    /// @brief Reads "<name>.backprop_net" (extension appended if missing) into a new Net.
    inline BackpropNet::Net load(const std::string& name) {
        const std::string path = withExtension(name);
        std::ifstream file(path);
        if(!file.is_open())
            throw std::runtime_error("BackpropNetReader::load: failed to open '" + path + "'");

        std::string magic;
        int version = 0;
        if(!(file >> magic >> version))
            throw std::runtime_error("BackpropNetReader::load: '" + path + "' is not a .backprop_net file");
        if(magic != "BACKPROP_NET")
            throw std::runtime_error("BackpropNetReader::load: '" + path + "' is not a .backprop_net file");
        if(version != FORMAT_VERSION)
            throw std::runtime_error("BackpropNetReader::load: unsupported version "
                                     + std::to_string(version) + " in '" + path + "'");

        size_t layerCount = 0;
        if(!(file >> layerCount))
            throw std::runtime_error("BackpropNetReader::load: missing layer count in '" + path + "'");

        BackpropNet::Net net;
        for(size_t i = 0; i < layerCount; i++) {
            std::string tag;
            if(!(file >> tag))
                throw std::runtime_error("BackpropNetReader::load: unexpected end of file at layer "
                                         + std::to_string(i));

            if(tag == "DEN") net.add_layer(loadDense(file, i));
            else if(tag == "ACT") net.add_layer(loadActivation(file, i));
            else throw std::runtime_error("BackpropNetReader::load: unknown layer type '" + tag
                                          + "' at layer " + std::to_string(i));
        }

        return net;
    }

}
