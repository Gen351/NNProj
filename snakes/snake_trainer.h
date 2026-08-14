#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <chrono>

#include "snake_environment.h"
#include "../nn_engine/simple_net/simple_net.h"

// Everything a net-driven run needs that the environment itself must not
// know about: move selection, tick caps, and the fitness/reward model.
// Kept as free functions so each piece is testable in isolation.
namespace SnakeTrainer {

struct RunResult {
    float fitness = 0.0f;
    int death_cause = 0;   // 0 none, 1 wall, 2 body, 3 win, 4 tick cap, 5 starved
    size_t game_ticks = 0;
    int score = 1;
};

// Argmax over the net's 3 outputs: 1 = turn left, 2 = straight, 3 = turn
// right, relative to the current heading. The environment only accepts 1..3,
// so there is no "reverse" action and a U-turn is impossible by construction.
inline int pickMove(const SnakesGame& game, const std::vector<float>& output) {
    int move = 2;
    float best = -1e30f;
    for(int i = 0; i < output.size(); ++i) {
        if(output[i] > best) { best = output[i]; move = i + 1; }
    }

    return move;
}

// Runs one full episode for a net. Resets the game first. The game object
// holds the final state afterwards (useful for screenshots/replays).
// Death causes: 1 wall, 2 body, 3 win, 4 tick cap, 5 starved (out-of-band).
inline RunResult run_net(SnakesGame& game, SimpleNet& net, bool display = false, int frame_delay_ms = 0) {
    RunResult result;
    game.reset();

    const size_t max_ticks_without_eat = game.board_size * game.board_size * 2;
    size_t ticks_since_eat = 0;

    if(display) game.display_game();

    while(!game.isGameOver() && game.game_ticks < game.max_game_tick) {
        const std::vector<float> state = game.getState();
        const std::vector<float> output = net.predict(state);

        const int move = pickMove(game, output);

        const SnakesGame::StepResult r = game.step(move);
        ticks_since_eat++;

        // Ate an apple? Later apples are worth more (harder to reach)
        if(r.ate) {
            result.fitness += 1200.0f;
            result.fitness -= ((float)max_ticks_without_eat / 1150.0f) * (ticks_since_eat);
            ticks_since_eat = 0;
        }

        if(game.snake.size == 1) {
            result.fitness -= ((float)max_ticks_without_eat / 1200.0f) * (ticks_since_eat);
        }

        if(display) {
            game.display_game(false);
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay_ms));
        }

        // Starvation: not eating for too long means it's going in circles;
        // end the run now so circling can never farm a tick-cap reward.
        if(ticks_since_eat > max_ticks_without_eat) {
            game.terminate(5);
            result.fitness -= ((float)max_ticks_without_eat * 1.5f);
            break;
        }
    }

    // Hit the tick cap: survived the run.
    if(!game.isGameOver() && game.game_ticks >= game.max_game_tick) {
        game.terminate(4);
        result.fitness -= ((float)max_ticks_without_eat * 0.4f) * (1.0f - (float)game.game_ticks / (float)game.max_game_tick);
    }

    // Death-time scaled penalties: dying early is punished much harder than
    // dying late, so long runs are strictly better.
    const int cause = game.deathCause();
    const float max_tick = (float)game.max_game_tick;
    if(cause == 1 || cause == 5) {
        result.fitness -= ((float)max_ticks_without_eat * 0.85f) * (1.0f - (float)game.game_ticks / max_tick);
    } else if(cause == 2) {
        result.fitness -= ((float)max_ticks_without_eat * 0.25f) * (1.0f - (float)game.game_ticks / max_tick);
    } else if(cause == 3) {
        result.fitness += 500000.0f; // won: board full
    }

    result.death_cause = cause;
    result.game_ticks = game.game_ticks;
    result.score = (int)game.snake.size;

    result.fitness += (0.9f * (float)game.game_ticks / max_tick);

    return result;
}

} // namespace SnakeTrainer
