#include <iostream>
#include <cstdlib>
#include <cmath>

#include "../nn_engine/simple_net/simple_net.h"
#include "../nn_engine/simple_net/simple_layer.h"
#include "../nn_engine/net_reader/simple_net_reader.h"
#include "./snake_environment.h"
#include "./snake_trainer.h"


int main(int argc, char* argv[]) {
    srand(std::time(nullptr));

    if(argc < 2) {
        std::cout << "\nUsage: test <model.simple_net> [speed_ms] [board_sz] <show?(0:1)>\n"
                  << "  -model.simple_net : path to a trained net relative to test.exe\n"
                  << "                    (e.g. ./trained_networks/run_19/best.simple_net)\n"
                  << "  -speed_ms         : ms per frame (default 80 = watchable)\n"
                  << "  -sggstd_board_sz  : board size\n"
				  << "  -show?            : show the AI solving the game <0 : 1>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const int speed_ms = (argc >= 3) ? std::atoi(argv[2]) : 80;
    const int suggested_board_size = (argc >= 4) ? std::atoi(argv[3]) : 35;
	const bool show_game = (argc >= 5) ? (std::atoi(argv[4]) == 0 ? false : true) : true;

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
    
    const int board_size = suggested_board_size;

    SnakesGame game(board_size);

    const SnakeTrainer::RunResult res = SnakeTrainer::run_net(game, net, show_game, speed_ms);

	game.display_game(false);
    std::cout << std::flush;

    // Show the model used
    std::cout << "\n\n\n\n> Model: " << model_path
              << "\n> Layers:\n";

    // 1. Extract Topology (Input size of first layer + output sizes of all layers)
    std::vector<size_t> topology;
    bool first_layer = true;
    for(const auto& layer : net.layers) {
        // Only count actual weight-bearing layers to determine size
        if (const auto* dense = dynamic_cast<const SimpleLayer*>(layer.get())) {
            if (first_layer) {
                topology.push_back(dense->weights.cols());
                first_layer = false;
            }
            topology.push_back(dense->weights.rows());
        }
    }

    // 2. Procedural ASCII Art Generator
    // Maps actual node count to visually aesthetic string length
    auto get_v = [](size_t size) -> int {
        if (size >= 40) return 18;
        if (size >= 30) return 16;
        if (size >= 25) return 14;
        if (size >= 20) return 12;
        if (size >= 16) return 10;
        if (size >= 10) return 8;
        if (size >= 4)  return 6;
        return 1;
    };

    const int center_offset = 20;

    for (size_t i = 0; i < topology.size(); ++i) {
        size_t s = topology[i];
        int v = get_v(s);
        
        // Draw Nodes
        std::string layer_str = "";
        if (v <= 1) {
            layer_str = "o";
        } else {
            layer_str.append(v / 2, 'o');
            layer_str += " ";
            layer_str.append(v / 2, 'o');
        }

        int pad_len = std::max(0, center_offset - (int)(layer_str.length() / 2));
        std::string pad(pad_len, ' ');
        
        std::cout << pad << layer_str << "... x " << s << "\n";

        // Draw Connections to next layer
        if (i < topology.size() - 1) {
            size_t next_s = topology[i + 1];
            int v_next = get_v(next_s);
            std::string conn_str = "";

            if (s == next_s) { // Same size
                conn_str = "|";
            } else if (s > next_s) { // Shrinking
                int slashes = std::max(1, std::max(v, v_next) / 2);
                conn_str.append(slashes, '\\');
                conn_str += " ";
                conn_str.append(slashes, '/');
            } else { // Expanding
                int slashes = std::max(1, std::max(v, v_next) / 2);
                conn_str.append(slashes, '/');
                conn_str += " ";
                conn_str.append(slashes, '\\');
            }

            int conn_pad_len = std::max(0, center_offset - (int)(conn_str.length() / 2));
            std::string conn_pad(conn_pad_len, ' ');
            
            std::cout << conn_pad << conn_str << "\n";
        }
    }

    std::cout << "\nFitness: " << res.fitness << "\n"
              << "Death Cause: " << res.death_cause
              << " (1=wall 2=body 3=win 4=tick cap 5=starved)\n"
              << "score: " << res.score - 1 << "\n";
              
    return 0;
}
