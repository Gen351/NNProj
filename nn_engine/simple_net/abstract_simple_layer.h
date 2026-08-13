#pragma once

#include<vector>
#include<memory>
#include<math.h>

enum class LayerType {
    ACT,
    LAY
};

class AbstractSimpleLayer {
protected:
    bool training = false;

public:
    virtual ~AbstractSimpleLayer() = default;

    virtual std::vector<float> forward(const std::vector<float>& previousInput) = 0;

    virtual std::unique_ptr<AbstractSimpleLayer> clone() const = 0;

    virtual std::string getType() const = 0;


    void train() {
        training = true;
    }
    void predict() {
        training = false;
    }
};