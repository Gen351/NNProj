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

// Argmax over the net's 4 outputs, never picking the exact reverse of the
// current direction (that would always be instant death). Falls back to the
// current direction, or Right when the snake hasn't moved yet.
// Throws if the net doesn't produce exactly 4 outputs (fail loud, not frozen).
inline int pickMove(const SnakesGame& game, const std::vector<float>& output) {
    if(output.size() != 4)
        throw std::runtime_error("SnakeTrainer::pickMove: net must output 4 values (got "
                                 + std::to_string(output.size()) + ")");

    const int rev = (game.snake.direction == 1) ? 3
                  : (game.snake.direction == 3) ? 1
                  : (game.snake.direction == 2) ? 4
                  : (game.snake.direction == 4) ? 2 : 0;
    int move = 0;
    float best = -1e30f;
    for(int i = 0; i < 4; ++i) {
        if(i + 1 == rev) continue;
        if(output[i] > best) { best = output[i]; move = i + 1; }
    }
    if(move == 0) move = (game.snake.direction == 0) ? 3 : game.snake.direction;
    return move;
}

// Stateless per-tick shaping reward for one live state:
//   + liveness       : being alive pays a little (float division, fixed)
//   + apple progress : reward actually getting closer to the apple (anti-circling)
//   + wall caution   : keep obstacles at a distance (min of the 8 rays)
// prev_dist must be the previous tick's head-to-apple distance / board_size.
inline float simpleFitness(const std::vector<float>& state,
                           size_t game_ticks,
                           size_t max_ticks_without_eat,
                           float prev_dist) {
    float f = 0.0f;

    // Liveness
    f -= 0.0005f * ((float)game_ticks / (float)max_ticks_without_eat);

    // Apple progress: state[3] = head-to-apple distance / board_size
    f += std::min(1.0f, std::max(0.0f, prev_dist - state[3])) * 1000.0f;

    // Wall/body caution: nearest of the 8 rays dominates
    float min_ray = state[10];
    for(int i = 1; i < 8; ++i) min_ray = std::min(min_ray, state[10 + i]);
    f += std::min(1.0f, min_ray) * 0.09f;

    return f;
}

// Runs one full episode for a net. Resets the game first. The game object
// holds the final state afterwards (useful for screenshots/replays).
// Death causes: 1 wall, 2 body, 3 win, 4 tick cap, 5 starved (out-of-band).
inline RunResult run_net(SnakesGame& game, SimpleNet& net, bool display = false, int frame_delay_ms = 0) {
    RunResult result;
    game.reset();

    const size_t max_ticks_without_eat = game.board_size * game.board_size * 2;
    size_t ticks_since_eat = 0;

    // Apple-progress baseline, normalized the same way as state[3].
    float prev_dist = game.getDistance(game.snake.head_x, game.snake.head_y,
                                       game.apple.x, game.apple.y) / (float)game.board_size;

    if(display) game.display_game();

    while(!game.isGameOver() && game.game_ticks < game.max_game_tick) {
        const std::vector<float> state = game.getState();
        const std::vector<float> output = net.predict(state);
        const int move = pickMove(game, output);

        const SnakesGame::StepResult r = game.step(move);
        ticks_since_eat++;

        // Ate an apple? Later apples are worth more (harder to reach)
        if(r.ate) {
            result.fitness += (3.0f * (float)game.snake.size) / (float)ticks_since_eat;
            ticks_since_eat = 0;
        }

        // Per-tick reward: this tick's state (progress + liveness + caution)
        result.fitness += simpleFitness(state, game.game_ticks, max_ticks_without_eat, prev_dist);
        // Rebaseline on the *current* apple so the distance just before an eat
        // isn't mistaken for progress toward the newly spawned one.
        prev_dist = game.getDistance(game.snake.head_x, game.snake.head_y,
                                     game.apple.x, game.apple.y) / (float)game.board_size;

        // Starvation: not eating for too long means it's going in circles;
        // end the run now so circling can never farm a tick-cap reward.
        if(ticks_since_eat > max_ticks_without_eat) {
            game.terminate(5);
            result.fitness -= 1000.0f;   // bounded, not unbounded (was -1.8M)
            break;
        }

        if(display) {
            game.display_game(false);
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay_ms));
        }
    }

    // Hit the tick cap: survived the run.
    if(!game.isGameOver() && game.game_ticks >= game.max_game_tick) {
        game.terminate(4);
        result.fitness += 9999999.0f; // A man can wish
    }

    // Death-time scaled penalties: dying early is punished much harder than
    // dying late, so long runs are strictly better.
    const int cause = game.deathCause();
    const float max_tick = (float)game.max_game_tick;
    if(cause == 1 || cause == 5) {
        result.fitness -= 2000.0f * (1.0f - (float)game.game_ticks / max_tick);
    } else if(cause == 2) {
        result.fitness -= 1000.0f * (1.0f - (float)game.game_ticks / max_tick);
    } else if(cause == 3) {
        result.fitness += 5000.0f; // won: board full
    }

    result.death_cause = cause;
    result.game_ticks = game.game_ticks;
    result.score = (int)game.snake.size;
    return result;
}

} // namespace SnakeTrainer
