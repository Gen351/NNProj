#pragma once

#include<vector>
#include<math.h>
#include<fstream>

enum class LayerType {
    ACT,
    LAY
};

class AbstractSimpleLayer {

protected:
    bool training = false;

public:
    virtual std::vector<float> forward(const std::vector<float>& previousInput) = 0;

    virtual void save(std::ofstream& file) const = 0;
    virtual void load(std::ofstream& file) const = 0;
    virtual std::string getType() const = 0;

    void train() {
        training = true;
    }
    void predict() {
        training = false;
    }
};