#pragma once

#include<vector>
#include<algorithm>
#include<iostream>
#include<random>
#include<sstream>
#include<thread>
#include<chrono>

#ifdef _WIN32
#include<windows.h>
#endif

#include "../nn_engine/matrix_op/matrix.hpp"
#include "../nn_engine/simple_net/simple_net.h"

namespace SnakesColors {
    // 24-bit (true color) RGB values
    inline constexpr int BLACK_R = 0,   BLACK_G = 0,   BLACK_B = 0;
    inline constexpr int WHITE_R = 255, WHITE_G = 255, WHITE_B = 255;
    inline constexpr int HEAD_R = 0,    HEAD_G = 255,  HEAD_B = 0;
    inline constexpr int BODY_R = 0,    BODY_G = 128,  BODY_B = 0;
    inline constexpr int APPLE_R = 255, APPLE_G = 0,   APPLE_B = 0;
    inline constexpr int IND_R = 50,      IND_G = 50,      IND_B = 50;
    inline constexpr int IND_DARK_R = 30, IND_DARK_G = 30, IND_DARK_B = 30;
}

inline void enableVt() {
    static bool enabled = false;
    if(enabled) return;
    enabled = true;
#ifdef _WIN32
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if(GetConsoleMode(out, &mode))
        SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

struct SnakesGame {
    struct Snake {
        /** Direction: 
         * - not moving(start of the game): 0 !
         * - to the left: 1 <
         * - move up: 2 ^
         * - to the right: 3 >
         * - go down: 4 v
        */ 
        int direction;
        int prev_direction;

        int head_x, head_y, tail_x, tail_y;
        size_t size;

        Snake() : direction(0)
                , prev_direction(0)
                , size(1)
                , head_x(0)
                , head_y(0)
                , tail_x(0)
                , tail_y(0) {}
        Snake(int _head_x, int _head_y, int _tail_x, int _tail_y) 
                : direction(0)
                , prev_direction(0)
                , size(1)
                , head_x(_head_x)
                , head_y(_head_y)
                , tail_x(_tail_x)
                , tail_y(_tail_y) {}

        void setDirection(int newDir) {
            direction = (newDir <= 4 && newDir >= 0) ? newDir : 0;
        }
    };
    struct Apple {
        int x, y;
        Apple()
                : x(-1)
                , y(-1) {}
        Apple(int _x, int _y)
                : x(_x)
                , y(_y) {}
        void setPos(int _x, int _y) {
            x = _x; y = _y;
        }
    };


    size_t game_ticks;
    // Set by run_net; normalizes state[4] into [0,1]
    size_t max_game_tick;
    // Square board
    size_t board_size;
    // to make this simple now, use a 2d matrix instead of a single vector
    // optimize later
    // board[i][j] = 1 if the snakes body is there
    Matrix<int> board;
    
    Snake snake;
    Apple apple;

    SnakesGame()
            : game_ticks(0)
            , max_game_tick(1)
            , board_size(10)
            , board(10, 10, 0) {
        int mid = board_size / 2;
        // Put the snake in the middle
        board[mid][mid] = 1;
        snake = Snake(mid, mid, mid, mid);
    }

    SnakesGame(size_t _board_size)
            : game_ticks(0)
            , max_game_tick(1)
            , board_size(_board_size)
            , board(_board_size, _board_size, 0) {
        int mid = board_size / 2;
        // Put the snake in the middle
        board[mid][mid] = 1;
        snake = Snake(mid, mid, mid, mid);
    }

    // Places the apple on a random empty cell. Never inside the body.
    // Apples in front of the head (1-2 cells ahead) are very unlikely,
    // so the snake isn't handed free apples.
    // Returns false when the board is full (win).
    bool spawnApple() {
        static thread_local std::mt19937 engine(std::random_device{}());
        static std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        const float ahead_spawn_chance = 0.005f; // 0.5%: apple may appear in the snake's path

        // Cells the head is facing (within the board, currently empty)
        std::vector<size_t> ahead;
        if(snake.direction != 0) {
            const int dx[5] = {0, -1, 0, 1, 0};
            const int dy[5] = {0, 0, -1, 0, 1};
            for(int steps = 1; steps <= 2; steps++) {
                const int ax = snake.head_x + dx[snake.direction] * steps;
                const int ay = snake.head_y + dy[snake.direction] * steps;
                if(ax < 0 || ax >= (int)board_size || ay < 0 || ay >= (int)board_size) continue;
                const size_t idx = (size_t)ay * board_size + (size_t)ax;
                if(board.data[idx] == 0) ahead.push_back(idx);
            }
        }

        std::vector<size_t> empty;
        for(size_t i = 0; i < board.data.size(); i++) {
            if(board.data[i] != 0) continue;
            if(std::find(ahead.begin(), ahead.end(), i) != ahead.end()) continue;
            empty.push_back(i);
        }
        if(empty.empty() && ahead.empty()) { apple = Apple(); return false; }

        if(!ahead.empty() && coin(engine) < ahead_spawn_chance) {
            // rare case: apple spawns right in the snake's path anyway
            std::uniform_int_distribution<size_t> dist(0, ahead.size() - 1);
            const size_t cell = ahead[dist(engine)];
            apple.setPos((int)(cell % board_size), (int)(cell / board_size));
            return true;
        }

        std::vector<size_t>& pool = empty.empty() ? ahead : empty;
        std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
        const size_t cell = pool[dist(engine)];
        apple.setPos((int)(cell % board_size), (int)(cell / board_size));
        return true;
    }


    float getDistance(int h_x, int h_y, int a_x, int a_y) {
        int x_delta = (h_x - a_x);
        int y_delta = (h_y - a_y);

        float dist = float(x_delta * x_delta) + float(y_delta * y_delta);

        return std::sqrt(dist);
    }

    float rayDistance(int dx, int dy) const {
        for (int s = 1; ; ++s) {
            int x = snake.head_x + dx * s, y = snake.head_y + dy * s;
            if (x < 0 || x >= (int)board_size || y < 0 || y >= (int)board_size) return (float)(s - 1);
            if (board(y, x) == 1) return (float)(s - 1);   // body (incl. tail nuance)
        }
    }

    int update() {
        if(snake.direction != 0) {
            // Head move to direction
            int new_head_x = snake.head_x + (snake.direction == 1 ? -1 : snake.direction == 3 ? 1 : 0);
            int new_head_y = snake.head_y + (snake.direction == 2 ? -1 : snake.direction == 4 ? 1 : 0);

            // Check 180-degree turn (reverse direction) - instant death
            if (snake.prev_direction != 0) {
                bool is_reverse = (snake.direction == 1 && snake.prev_direction == 3) ||
                                  (snake.direction == 3 && snake.prev_direction == 1) ||
                                  (snake.direction == 2 && snake.prev_direction == 4) ||
                                  (snake.direction == 4 && snake.prev_direction == 2);
                if (is_reverse) {
                    return 2; // collided with body (self)
                }
            }

            // Checks
            if(new_head_x < 0 || new_head_x >= (int)board_size
            || new_head_y < 0 || new_head_y >= (int)board_size) {
                // collided with wall
                return 1;
            }
            
            // Body collision check
            // For size == 2, the tail IS the neck, so moving into it is a 180-turn collision
            // For size > 2, tail moves away, so it's safe unless it's not the tail
            bool is_tail = (new_head_x == snake.tail_x && new_head_y == snake.tail_y);
            bool tail_moves = (snake.size > 2); // tail vacates for size > 2
            
            if(board(new_head_y, new_head_x) == 1 && !(is_tail && tail_moves)) {
                // collided with body
                return 2;
            }

            // update
            bool eating = (apple.x != -1 && new_head_x == apple.x && new_head_y == apple.y);
            int old_head_x = snake.head_x, old_head_y = snake.head_y;
            int old_tail_x = snake.tail_x, old_tail_y = snake.tail_y;

            board(new_head_y, new_head_x) = 1;
            snake.head_x = new_head_x;
            snake.head_y = new_head_y;
            
            // Store prev_direction after successful move
            snake.prev_direction = snake.direction;

            if(eating) {
                snake.size++;
                if(!spawnApple()) { return 3; }
            }
            else if(snake.size == 1) {
                // single cell: vacate it, tail stays with the head
                board(old_head_y, old_head_x) = 0;
                snake.tail_x = new_head_x;
                snake.tail_y = new_head_y;
            }
            else if(new_head_x == old_tail_x && new_head_y == old_tail_y) {
                // tail chase: the old head becomes the new tail, nothing is cleared
                snake.tail_x = old_head_x;
                snake.tail_y = old_head_y;
            }
            else {
                board(old_tail_y, old_tail_x) = 0;
                // new tail = the occupied neighbor of the old tail (excluding the new head)
                const int dx[4] = {1, -1, 0, 0};
                const int dy[4] = {0, 0, 1, -1};
                for(int i = 0; i < 4; i++) {
                    int nx = old_tail_x + dx[i];
                    int ny = old_tail_y + dy[i];
                    if(nx < 0 || nx >= (int)board_size || ny < 0 || ny >= (int)board_size) continue;
                    if(nx == new_head_x && ny == new_head_y) continue;
                    if(board(ny, nx) == 1) {
                        snake.tail_x = nx;
                        snake.tail_y = ny;
                        break;
                    }
                }
            }
        }

        return -1;
    }


    /**
     * Read user input from keys or mouse.
     * Update physics, positions, and AI logic.
     * Draw the new frame on the screen.
     * Repeat the cycle for the next frame.
     * 
     * Fixed Time Steps: 
     *  Run the simulation math at a fixed speed
     *      , like 60 times a second.
     *  Draw the graphics as fast as the monitor can show them.
     *  Separate math updates from display frames to avoid stutter.
     */

    // The displays are optional because you can display    

    // Retun the cause of death:
        // collided with wall(1): * 0.5%
        // collided with body(2): * 0.55%
        // board full(3): +100% (win)
    int run(bool display=false, int frame_delay_ms=0) {
        int game_over_value = -1;
        if(display) {display_game(true);}

        while(game_over_value == -1) {
            // Input
            
            // Simulate
            game_over_value = update();

            game_ticks++;
            if(display) {
                display_game(false);    // overwrite the previous frame, no flicker
                std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay_ms));
            }   
        }

        return game_over_value;
    }

    /** Network architecture should be:
     * - Input Layer~~~~~~~~: Input [game.getState().size()] 
     * ~~~~~~~~~~~~~~~~~~~~~| Output [User_Def_N-1]
     * 
     * - Hidden Layer_N~~~~~: Input [User_Def_N-1] 
     * ~~~~~~~~~~~~~~~~~~~~~| Output [User_Def_N]
     * 
     * - Output Layer~~~~~~~: Input [User_Def_N] 
     * ~~~~~~~~~~~~~~~~~~~~~| Output [4] (directions)
     */
    float run_net(bool display, SimpleNet& net, int frame_delay_ms=0) {
        float fitness = 0.0f;
        // For early stopping
        // Medium -> Expert level of difficulty (higher = longer runs, but can have higher scores)
        int level = 30;
        int max_game_tick = board_size * board_size * level;
        size_t prev_size = 1;
        // Hunger: if the snake doesn't eat for this long, it's stuck (circling)
        // and the run is cut short so it can't farm liveness up to the tick cap.
        const size_t max_ticks_without_eat = board_size * board_size * 2;
        size_t ticks_since_eat = 0;
        max_game_tick = (size_t)(board_size * board_size * level);

        // Fresh generation: reset the game state
        board = Matrix<int>(board_size, board_size, 0);
        const int mid = (int)(board_size / 2);
        snake = Snake(mid, mid, mid, mid);
        spawnApple();
        game_ticks = 0;
        int game_over_value = -1;
        // Distance to the apple, normalized the same way as state[3]
        float prev_dist = getDistance(snake.head_x, snake.head_y, apple.x, apple.y) / board_size;
        
        // display_game() (optional)
        if(display) {display_game();}
        
        while(game_over_value == -1 && game_ticks <= max_game_tick) {
            // Input
            std::vector<float> state = getState();
            std::vector<float> output = net.predict(state);
            
            // == Network's move ===== //
            // Never let the net pick the exact reverse of the current direction
            // (a hard -1e9 for that made every game die before accumulating
            // any fitness); fall back to its best other move instead.
            const int rev = (snake.direction == 1) ? 3
                          : (snake.direction == 3) ? 1
                          : (snake.direction == 2) ? 4
                          : (snake.direction == 4) ? 2 : 0;
            int move = 0;
            float highest_dist = -999999999999.0f; // I don't know flaot max
            for(int i = 0; i < output.size(); i++) {
                const int cand = i + 1;
                if(cand == rev) continue;
                if(output[i] > highest_dist) {
                    move = cand;
                    highest_dist = output[i];
                }
            }
            if(move == 0) move = (snake.direction == 0) ? 3 : snake.direction;
            snake.setDirection(move);
            // ======================= //
            
            // Simulate
            game_over_value = update();

            // update gameticks
            game_ticks++;
            ticks_since_eat++;

            // Ate an apple? Later apples are worth more (harder to reach)
            if(snake.size > prev_size) {
                fitness += 50.0f * (float)snake.size;
                prev_size = snake.size;
                ticks_since_eat = 0;
            }

            // Per-tick reward: reuse this tick's state (progress + liveness + caution)
            fitness += simpleFitness(state, board_size, prev_dist, max_ticks_without_eat);
            // Rebaseline the progress term on the *current* apple, so the
            // distance just before an eat isn't mistaken for progress toward
            // the newly spawned one.
            prev_dist = getDistance(snake.head_x, snake.head_y, apple.x, apple.y) / board_size;

            // Starvation: not eating for too long means it's going in circles;
            // end the run now so circling can never score a tick-cap reward.
            if(ticks_since_eat > max_ticks_without_eat) {
                game_over_value = 5; // starved
                break;
            }
            
            // display_game() (optional)
            if(display) {
                display_game(false);    // overwrite the previous frame, no flicker
                std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay_ms));
            }   
        }

        if(game_ticks >= max_game_tick && game_over_value == -1) {
            game_over_value = 4; // hit the tick cap (survived the run)
        }

        // Death-time scaled penalties: dying early is punished much harder
        // than dying late, so long runs are strictly better.
        if(game_over_value == 1 || game_over_value == 5) {
            // wall hit, or starved (circled without eating)
            fitness -= 2000.0f * (1.0f - (float)game_ticks / (float)max_game_tick);
        } else if(game_over_value == 2) {
            fitness -= 1000.0f * (1.0f - (float)game_ticks / (float)max_game_tick);
        } else if(game_over_value == 3) {
            fitness += 5000.0f; // won: board full
        }

        return fitness;
    }

    // Stateless per-tick shaping. Returns the reward for the current state:
    //   + liveness          : being alive at all pays a little
    //   + apple progress    : only reward actually getting closer to the apple
    //                         (anti-circling: a parked snake earns 0 here)
    //   + wall caution      : keep obstacles at a distance (min of the 8 rays)
    // prev_dist must be the previous tick's head-to-apple distance, normalized
    // the same way as state[3] (i.e. / board_size).
    float simpleFitness(const std::vector<float>& game_state, const size_t board_size, float prev_dist, int max_ticks_without_eat) {
        float finalFitness = 0.0f;

        // Liveness
        finalFitness += 0.005f * (game_ticks / max_ticks_without_eat);

        // Apple progress: state[3] = head-to-apple distance / board_size
        finalFitness += std::min(1.0f, std::max(0.0f, prev_dist - game_state[3])) * 6.0f;

        // Wall/body caution: nearest of the 8 rays dominates
        float min_ray = game_state[10];
        for(int i = 1; i < 8; i++) min_ray = std::min(min_ray, game_state[10 + i]);
        finalFitness += std::min(1.0f, min_ray) * 0.05f;

        return finalFitness;
    }


    // The AI's "eyes"
    std::vector<float> getState() {
        size_t state_size = getStateSize();

        // Get direction of intention
        float dir_head_x = snake.head_x;
        float dir_head_y = snake.head_y;
        switch(snake.direction) {
            case 1: dir_head_x --; break;
            case 2: dir_head_y --; break;
            case 3: dir_head_x ++; break;
            case 4: dir_head_y ++; break;
        };
        
        float max_dist = board_size;
        float dist_from_dir_to_apple = getDistance(dir_head_x, dir_head_y, apple.x, apple.y);
        float dist_from_head_to_apple = getDistance(snake.head_x, snake.head_y, apple.x, apple.y);

        std::vector<float> state(state_size);
        
        state[0] = float(snake.direction) / 4.0f;
        state[1] = float(snake.size) / float(board_size * board_size);
        state[2] = dist_from_dir_to_apple / max_dist;
        state[3] = dist_from_head_to_apple / max_dist;
        state[4] = float(game_ticks) / float(max_game_tick);

        // Distances from the wall
        if(snake.direction == 1) {
            state[5] = snake.head_x;
            state[6] = board_size - snake.head_y - 1;
            state[7] = snake.head_y;
        } else if(snake.direction == 2) {
            state[5] = snake.head_y;
            state[6] = snake.head_x;
            state[7] = board_size - snake.head_x - 1;
        } else if(snake.direction == 3) {
            state[5] = board_size - snake.head_x - 1;
            state[6] = snake.head_y;
            state[7] = board_size - snake.head_y - 1;
        } else if(snake.direction == 4) {
            state[6] = board_size - snake.head_y - 1;
            state[5] = board_size - snake.head_x - 1;
            state[7] = snake.head_x;
        }
        
        // --- 1. Compute Forward (f) Vector based on snake.direction ---
        int f_dx = 0, f_dy = 0;
        switch(snake.direction) {
            case 1: f_dx = -1; f_dy =  0; break; // Left
            case 2: f_dx =  0; f_dy = -1; break; // Up
            case 3: f_dx =  1; f_dy =  0; break; // Right
            case 4: f_dx =  0; f_dy =  1; break; // Down
            default: f_dx = 0; f_dy = -1; break; // Default to Up if stationary (dir 0)
        }

        // --- 2. Compute Right (r) Vector by rotating Forward 90° Clockwise ---
        int r_dx = -f_dy;
        int r_dy =  f_dx;

        // Apple offset in the snake's own frame (same frame as rays/walls):
        // state[8] = ahead(+) / behind(-), state[9] = right(+) / left(-).
        // One "wall ahead -> turn toward apple" rule now works at all four walls.
        state[8] = ((apple.x - snake.head_x) * f_dx + (apple.y - snake.head_y) * f_dy) / max_dist;
        state[9] = ((apple.x - snake.head_x) * r_dx + (apple.y - snake.head_y) * r_dy) / max_dist;

        // --- 3. Build 8 Relative Directions ---
        int rel_dirs[8][2] = {
            { f_dx,          f_dy          }, // Ray 0: Forward
            { f_dx + r_dx,   f_dy + r_dy   }, // Ray 1: Forward-Right
            { r_dx,          r_dy          }, // Ray 2: Right
            {-f_dx + r_dx,  -f_dy + r_dy   }, // Ray 3: Backward-Right
            {-f_dx,         -f_dy          }, // Ray 4: Backward
            {-f_dx - r_dx,  -f_dy - r_dy   }, // Ray 5: Backward-Left
            {-r_dx,         -r_dy          }, // Ray 6: Left
            { f_dx - r_dx,   f_dy - r_dy   }  // Ray 7: Forward-Left
        };

        // --- 4. Populate Raycast Distances ---
        for (int i = 0; i < 8; i++) {
            state[10 + i] = rayDistance(rel_dirs[i][0], rel_dirs[i][1]) / max_dist; 
        }

        return state;
    }

    // Screenshot only: renders the current game state once.
    void display_game(bool clear_scr=true) {
        enableVt();

        // Buffer the whole frame, then write it in one shot
        std::ostringstream frame;
        if(clear_scr) {frame << "\x1b[H\x1b[2J";}   // first frame: clear screen
        else {frame << "\x1b[H";}                   // later frames: overwrite in place

        frame << "\n\n\n\n";
        // Cells the snake is heading toward (beside the head, and one more ahead)
        int dir_indication_x = snake.head_x;
        int dir_indication_y = snake.head_y;
        int dir_indication2_x = dir_indication_x;
        int dir_indication2_y = dir_indication_y;
        switch(snake.direction) {
            case 0: break;
            case 1: dir_indication_x--; dir_indication2_x -= 2; break;
            case 2: dir_indication_y--; dir_indication2_y -= 2; break;
            case 3: dir_indication_x++; dir_indication2_x += 2; break;
            case 4: dir_indication_y++; dir_indication2_y += 2; break;
        }
        // Clamp overflows so the indicators stay on the board
        if(dir_indication_x < 0) dir_indication_x = 0;
        if(dir_indication_x >= (int)board_size) dir_indication_x = (int)board_size - 1;
        if(dir_indication_y < 0) dir_indication_y = 0;
        if(dir_indication_y >= (int)board_size) dir_indication_y = (int)board_size - 1;
        if(dir_indication2_x < 0) dir_indication2_x = 0;
        if(dir_indication2_x >= (int)board_size) dir_indication2_x = (int)board_size - 1;
        if(dir_indication2_y < 0) dir_indication2_y = 0;
        if(dir_indication2_y >= (int)board_size) dir_indication2_y = (int)board_size - 1;

        const size_t n = board_size + 2;
        for(size_t row = 0; row < n; row++) {
            for(size_t col = 0; col < n; col++) {
                int r = SnakesColors::BLACK_R;
                int g = SnakesColors::BLACK_G;
                int b = SnakesColors::BLACK_B;

                if(row == 0 || col == 0 || row == n - 1 || col == n - 1) {
                    // Border frame around the board
                    r = SnakesColors::WHITE_R;
                    g = SnakesColors::WHITE_G;
                    b = SnakesColors::WHITE_B;
                }
                else {
                    size_t br = row - 1;
                    size_t bc = col - 1;
                    if(bc == (size_t)snake.head_x && br == (size_t)snake.head_y) {
                        r = SnakesColors::HEAD_R;
                        g = SnakesColors::HEAD_G;
                        b = SnakesColors::HEAD_B;
                    }
                    else if(apple.x != -1 && bc == (size_t)apple.x && br == (size_t)apple.y) {
                        r = SnakesColors::APPLE_R;
                        g = SnakesColors::APPLE_G;
                        b = SnakesColors::APPLE_B;
                    }
                    else if(board(br, bc) == 1) {
                        r = SnakesColors::BODY_R;
                        g = SnakesColors::BODY_G;
                        b = SnakesColors::BODY_B;
                    }
                    else if(bc == (size_t)dir_indication_x && br == (size_t)dir_indication_y) {
                        r = SnakesColors::IND_R;
                        g = SnakesColors::IND_G;
                        b = SnakesColors::IND_B;
                    }
                    else if(bc == (size_t)dir_indication2_x && br == (size_t)dir_indication2_y) {
                        r = SnakesColors::IND_DARK_R;
                        g = SnakesColors::IND_DARK_G;
                        b = SnakesColors::IND_DARK_B;
                    }
                }

                frame << "\x1b[48;2;" << r << ";" << g << ";" << b << "m  \x1b[0m";
            }
            frame << "\n";
        }
        std::cout << frame.str() << std::flush;
    }



    size_t getStateSize() {
        size_t state_size = // snake.direction 0
                            1 
                            // snake.size 1
                            + 1
                            // distance from direction of intention to apple
                            + 1
                            // distance from head to apple 8
                            + 1
                            // game_ticks / max_game_tick 9
                            + 1
                            // distance from the wall front(to the direction)
                            + 1
                            // distance from the wall left
                            + 1
                            // distance from the wall right
                            + 1
                            // angle direction x to apple
                            + 1
                            // angle direction y to apple
                            + 1
                            // 8 Ray casts
                            + 8;

        return state_size;
    }
};