#pragma once

#include <vector>
#include <deque>
#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <chrono>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

#include "../nn_engine/matrix_op/matrix.hpp"

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

// Pure game environment. It knows nothing about neural networks, fitness,
// or training loops -- callers step() it and read getState() themselves.
// Move encoding: step() takes a relative move (1 = turn left, 2 = straight,
// 3 = turn right) relative to the snake's current heading. The heading is
// tracked internally (snake.direction) as an absolute direction:
// 1 = left, 2 = up, 3 = right, 4 = down.
struct SnakesGame {
    struct Pos {
        int x, y;
        bool operator==(const Pos& o) const { return x == o.x && y == o.y; }
    };

    // Death causes returned by step() / run() / deathCause():
    //   0 = still alive, 1 = wall, 2 = body, 3 = win (board full).
    // The trainer adds its own out-of-band causes (tick cap, starvation)
    // via terminate().
    struct StepResult {
        int cause = 0;
        bool ate = false;
    };

    struct Snake {
        int direction = 0;       // 1..4, 0 = not moving yet
        int prev_direction = 0;
        std::deque<Pos> body;    // front = head; single source of truth
        size_t size = 1;         // == body.size(), kept in sync
        int head_x = 0, head_y = 0;   // == body.front(), kept in sync
        int tail_x = 0, tail_y = 0;   // == body.back(), kept in sync

        Snake(int board_size) {
            head_x = rand() % (board_size - 1);
            head_y = rand() % (board_size - 1);
            tail_x = head_x;
            tail_y = head_y;
        }

        Snake() = default;
    };

    struct Apple {
        int x = -1, y = -1;
    };

    size_t game_ticks = 0;       // completed (non-terminal) moves only
    size_t max_game_tick;        // tick cap / state[4] normalizer (set by reset)
    size_t board_size;
    Matrix<int> board;           // 1 = body cell, always matches snake.body
    Snake snake;
    Apple apple;

    // Constructs a fresh game on a square board. seed lets you replay a run
    // deterministically.
    SnakesGame() : SnakesGame(10) {
        snake = Snake(10);
    }

    SnakesGame(size_t _board_size, unsigned seed = std::random_device{}())
        : board_size(_board_size)
        , board(_board_size, _board_size, 0)
        , rng(seed) {
        snake = Snake(board_size);
        reset();
    }

    // Fresh game: centered size-1 snake + a randomly placed apple.
    void reset() {
        game_ticks = 0;
        game_over = 0;
        max_game_tick = board_size * board_size * 30;
        snake = Snake(board_size);
        const int mid = (int)(board_size / 2);
        snake.body = { Pos{snake.head_x, snake.head_y} };
        syncGeometry();
        apple = Apple();
        spawnApple();
    }

    // Test helper: replace the snake body (front = head), repaint the board,
    // and re-sync the cached head/tail/size fields.
    void setBody(const std::deque<Pos>& new_body) {
        snake.body = new_body;
        game_over = 0;
        game_ticks = 0;
        syncGeometry();
    }

    bool isGameOver() const { return game_over != 0; }
    int deathCause() const { return game_over; }
    int score() const { return (int)snake.size; }

    // End the episode without a step (trainer-side causes: 4 = tick cap,
    // 5 = starvation). No-op once the game is already over.
    void terminate(int cause) {
        if(game_over == 0) game_over = cause;
    }

    // Execute one relative move: 1 = turn left, 2 = straight, 3 = turn right,
    // relative to the current heading. The heading is tracked internally as an
    // absolute direction (1 = left,  = up, 3 = right, 4 = down), so a
    // reversal is impossible by construction. Throws on an invalid move so a
    // bad caller fails loudly instead of silently freezing the snake.
    // Returns cause 0 (alive), 1 (wall), 2 (body), 3 (win) + whether it ate.
    StepResult step(int move) {
        if(move < 1 || move > 3)
            throw std::runtime_error("SnakesGame::step: invalid move " + std::to_string(move));
        if(game_over != 0) return {game_over, false};

        // Resolve the relative input against the current heading. Before the
        // snake has moved (direction 0) it is treated as facing up, matching
        // getState().
        const int base = (snake.direction == 0) ? 2 : snake.direction;
        const int dir = (move == 1) ? (base == 1 ? 4 : base - 1)
                      : (move == 3) ? (base == 4 ? 1 : base + 1)
                      : base;

        const int dx[5] = {0, -1, 0, 1, 0};
        const int dy[5] = {0, 0, -1, 0, 1};
        const int nx = snake.head_x + dx[dir];
        const int ny = snake.head_y + dy[dir];

        // Wall
        if(nx < 0 || nx >= (int)board_size || ny < 0 || ny >= (int)board_size) {
            game_over = 1;
            return {1, false};
        }

        const bool eating = (apple.x == nx && apple.y == ny);

        // Body collision. The tail cell vacates this tick, so the head may
        // enter it -- unless the snake is eating (then the tail stays).
        // This one uniform rule handles tail-chase (size 2) and size-1
        // reversal with no special cases.
        for(size_t i = 0; i < snake.body.size(); ++i) {
            if(snake.body[i].x == nx && snake.body[i].y == ny) {
                const bool is_tail = (i == snake.body.size() - 1);
                if(eating || !is_tail) {
                    game_over = 2;
                    return {2, false};
                }
                break;
            }
        }

        // Commit the move.
        snake.prev_direction = snake.direction;
        snake.direction = dir;
        snake.body.push_front(Pos{nx, ny});
        if(!eating) snake.body.pop_back();
        syncGeometry();

        game_ticks++;
        if(eating) {
            if(!spawnApple()) { game_over = 3; return {3, true}; }
            return {0, true};
        }
        return {0, false};
    }

    // Places the apple on a random empty cell, never inside the body.
    // Apples 1-2 cells ahead of the head are very unlikely, so the snake
    // isn't handed free apples. Returns false when the board is full (win).
    bool spawnApple() {
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

        std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        if(!ahead.empty() && coin(rng) < ahead_spawn_chance) {
            std::uniform_int_distribution<size_t> dist(0, ahead.size() - 1);
            const size_t cell = ahead[dist(rng)];
            apple.x = (int)(cell % board_size);
            apple.y = (int)(cell / board_size);
            return true;
        }

        std::vector<size_t>& pool = empty.empty() ? ahead : empty;
        std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
        const size_t cell = pool[dist(rng)];
        apple.x = (int)(cell % board_size);
        apple.y = (int)(cell / board_size);
        return true;
    }

    float getDistance(int h_x, int h_y, int a_x, int a_y) const {
        const int x_delta = (h_x - a_x);
        const int y_delta = (h_y - a_y);
        return std::sqrt((float)(x_delta * x_delta) + (float)(y_delta * y_delta));
    }

    // Free cells from the head to the nearest wall along (dx, dy).
    float wallDistance(int dx, int dy) const {
        float d = 0.0f;
        while(true) {
            const int x = snake.head_x + dx * ((int)d + 1);
            const int y = snake.head_y + dy * ((int)d + 1);
            if(x < 0 || x >= (int)board_size || y < 0 || y >= (int)board_size) return d;
            d += 1.0f;
        }
    }

    // 1. Wall Distance: Steps out until hitting the outer board boundary
    float rayWallDistance(int hx, int hy, int dx, int dy) const {
        if (dx == 0 && dy == 0) return 0.0f;
        for(int s = 1; ; ++s) {
            const int x = hx + dx * s, y = hy + dy * s;
            if(x < 0 || x >= (int)board_size || y < 0 || y >= (int)board_size) {
                return (float)(s - 1);
            }
        }
    }

    // 2. Body Distance: Steps out until hitting the snake's body or a wall
    float rayBodyDistance(int hx, int hy, int dx, int dy) const {
        if (dx == 0 && dy == 0) return 0.0f;
        for(int s = 1; ; ++s) {
            const int x = hx + dx * s, y = hy + dy * s;
            // If it hits a wall before finding a body, return max distance (clear)
            if(x < 0 || x >= (int)board_size || y < 0 || y >= (int)board_size) {
                return (float)(s - 1);
            }
            // If it hits a body segment (assuming 1 represents the body)
            if(board(y, x) == 1) {
                return (float)(s - 1);
            }
        }
    }

    // 3. Food Direction: Returns 1.0 if the apple lies along this specific ray vector, 0.0 otherwise
    float rayFoodDistance(int hx, int hy, int dx, int dy) const {
        if (dx == 0 && dy == 0) return 0.0f;
        for(int s = 1; ; ++s) {
            const int x = hx + dx * s, y = hy + dy * s;
            // Stop searching if out of bounds
            if(x < 0 || x >= (int)board_size || y < 0 || y >= (int)board_size) {
                break;
            }
            // Check if apple coordinates match this ray step
            if(x == apple.x && y == apple.y) {
                return 1.0f;
            }
        }
        return 0.0f;
    }

    // 20 hand-crafted features. Layout is compatible with the old game so
    std::vector<float> getState() const {
        std::vector<float> state(24);
        const float max_dist = (float)board_size;
        const int hx = snake.head_x, hy = snake.head_y;

        // Snake's forward vector
        int f_dx, f_dy;
        switch(snake.direction) {
            case 1: f_dx = -1; f_dy =  0; break; // left
            case 2: f_dx =  0; f_dy = -1; break; // up
            case 3: f_dx =  1; f_dy =  0; break; // right
            case 4: f_dx =  0; f_dy =  1; break; // down
            default: f_dx = 0; f_dy = -1; break; // not moving yet -> face up
        }
        const int r_dx = -f_dy, r_dy = f_dx; // snake's right vector

        // 8 relative directions around the head
        const int rel_dirs[8][2] = {
            { f_dx,        f_dy        }, // 0: forward
            { f_dx + r_dx, f_dy + r_dy }, // 1: forward-right
            { r_dx,        r_dy        }, // 2: right
            {-f_dx + r_dx, -f_dy + r_dy }, // 3: backward-right
            {-f_dx,       -f_dy        }, // 4: backward
            {-f_dx - r_dx, -f_dy - r_dy }, // 5: backward-left
            {-r_dx,       -r_dy        }, // 6: left
            { f_dx - r_dx, f_dy - r_dy }  // 7: forward-left
        };

        for(int i = 0; i < 8; ++i) {
            int dx = rel_dirs[i][0];
            int dy = rel_dirs[i][1];

            // 1. Wall Distance (Indices 0 - 7)
            state[i] = rayWallDistance(hx, hy, dx, dy) / max_dist;

            // 2. Body Distance (Indices 8 - 15)
            state[8 + i] = rayBodyDistance(hx, hy, dx, dy) / max_dist;

            // 3. Food Direction / Distance (Indices 16 - 23)
            state[16 + i] = rayFoodDistance(hx, hy, dx, dy) / max_dist;
        }

        return state;
    }

    // Must stay 20: existing .simple_net files validate their input size
    // against this at load time.
    size_t getStateSize() const { return 24; }

    // Manual/testing play: keep moving straight in the current direction
    // until death.
    int run(bool display = false, int frame_delay_ms = 0) {
        if(display) display_game(true);
        while(game_over == 0) {
            if(snake.direction == 0) break;
            const StepResult r = step(2);
            if(display) {
                display_game(false); // overwrite the previous frame, no flicker
                std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay_ms));
            }
            if(r.cause != 0) return r.cause;
        }
        return game_over;
    }

    // Screenshot only: renders the current game state once.
    void display_game(bool clear_scr = true) {
        enableVt();

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

private:
    int game_over = 0;
    std::mt19937 rng;

    // Rebuild the board from snake.body and refresh the cached head/tail/size.
    void syncGeometry() {
        if(snake.body.empty()) return;
        snake.head_x = snake.body.front().x;
        snake.head_y = snake.body.front().y;
        snake.tail_x = snake.body.back().x;
        snake.tail_y = snake.body.back().y;
        snake.size = snake.body.size();
        board = Matrix<int>(board_size, board_size, 0);
        for(const auto& p : snake.body) board(p.y, p.x) = 1;
    }
};
