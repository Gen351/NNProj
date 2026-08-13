#include <iostream>
#include <cstdlib>
#include <cmath>

#include "../nn_engine/simple_net/simple_net.h"
#include "../nn_engine/simple_net/simple_layer.h"
#include "../nn_engine/net_reader/simple_net_reader.h"
#include "./snake_environment.h"
#include "./snake_trainer.h"


int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cout << "Usage: test <model.simple_net> [speed_ms]\n"
                  << "  model.simple_net : path to a trained net (e.g. ./trained_networks/run_19/best.simple_net)\n"
                  << "  speed_ms         : ms per frame (default 80 = watchable)\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const int speed_ms = (argc >= 3) ? std::atoi(argv[2]) : 80;

    SimpleNet net;
    try {
        net = SimpleNetReader::load(model_path);
    } catch(const std::exception& e) {
        std::cerr << "test: failed to load model: " << e.what() << "\n";
        return 1;
    }
    if(net.layers.empty()) {
        std::cerr << "test: model has no layers: " << model_path << "\n";
        return 1;
    }

    // Model input is the fixed 18-feature game state; the board size is no
    // longer encoded in the net, so it is hardcoded to match main.cpp.
    const SimpleLayer* first = dynamic_cast<const SimpleLayer*>(net.layers[0].get());
    if(!first) {
        std::cerr << "test: first layer is not a SimpleLayer\n";
        return 1;
    }
    
    const int board_size = 20;

    SnakesGame game(board_size);

    std::cout << "Playing " << board_size << "x" << board_size
              << " | model: " << model_path
              << " | speed_ms: " << speed_ms << "\n";

    const SnakeTrainer::RunResult res = SnakeTrainer::run_net(game, net, true, speed_ms);

    std::cout << "\nfitness: " << res.fitness << "\n"
              << "death cause: " << res.death_cause
              << " (1=wall 2=body 3=win 4=tick cap 5=starved)\n"
              << "score: " << res.score << "\n";
    return 0;
}
