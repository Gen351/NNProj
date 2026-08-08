#include <iostream>

#include "../nn_engine/simple_net/simple_net.h"
#include "./snakes_game.h"



int main(int argc, char* argv[]) {
    
    SnakesGame game(20);


    // == Network Variables ========= //
    const size_t input = game.getStateSize();
    const size_t output = 4;
    // ============================= //


    // Design network ============== //
    



    // ============================= //

    

    

    return 0;
}