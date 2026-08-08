#include <iostream>

#include "../nn_engine/simple_net/simple_net.h"
#include "./snakes_game.h"


float calculateFitness(const std::vector<float>& gameState);


int main(int argc, char* argv[]) {
    // Read User Args ============== //
    if(argc >= 2) {
        try {
            // Read some args
        } catch(...) {
            throw std::runtime_error("Error! idk error");
        }
    }
    // ============================= //


    // Train Variables ============= //
    const int board_size = 20;

    const int max_gen = 100;
    // Need threading for this, implement later
    // const int train_per_gen = 2;
    
    // fitness *= collision_punishment
    const float wall_collision_punishment = 0.5f;
    const float body_collision_punishment = 0.55f;
    
    // 0.3 = 30% | 0.2 = 20% ...
    const float mutation = 0.3f;
    // mutation delta
    const float mutation_alpha = 0.01f;
    
    float best_fitness = -99999.0f;
    SimpleNet bestNet;
    // ============================= //
    
    // Need to create now for getStateSize();
    SnakesGame game(board_size);
    
    // Network Variables =========== //
    const size_t input = game.getStateSize();
    const size_t output = 4;
    // ============================= //

    






    // Design network ============== //
    SimpleNet net;
    net.addLayer(std::make_unique<SimpleLayer>(input, 60));
    net.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::SIGM));
    
    net.addLayer(std::make_unique<SimpleLayer>(60, 30));
    net.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::LEAK));
    
    net.addLayer(std::make_unique<SimpleLayer>(30, output));
    net.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::RELU));
    // ============================= //



    // Train Loop ================== //
    int gen_count = 0;
    float curr_fitness = best_fitness;
    
    /**
     * No multitraining(threaded) run yet
     * sooo... one net per gen
     */
    SimpleNet trainNet;

    while(max_gen != -1 || gen_count < max_gen) {
        // Run the game
        game.run_net(false, trainNet, 0);

        // Mutate




        curr_fitness = calculateFitness(game.getState());
        if(curr_fitness > best_fitness) {
            bestNet = trainNet;
            best_fitness = curr_fitness;
        }

        // Save 
    }


    // ============================= //
    

    return 0;
}




float calculateFitness(const std::vector<float>& gameState) {

}