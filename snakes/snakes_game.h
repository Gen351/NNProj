#pragma once

#include<vector>
#include<iostream>

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

        int head_x, head_y, tail_x, tail_y;
        size_t size;

        Snake() : direction(0)
                , size(1)
                , head_x(0)
                , head_y(0)
                , tail_x(0)
                , tail_y(0) {}
        Snake(int _head_x, int _head_y, int _tail_x, int _tail_y) 
                : direction(0)
                , size(1)
                , head_x(_head_x)
                , head_y(_head_y)
                , tail_x(_tail_x)
                , tail_y(_tail_y) {}
    };
    struct Apple {
        int x, y;
        Apple()
                : x(-1)
                , y(-1) {}
        Apple(int _x, int _y)
                : x(_x)
                , y(_y) {}
    };


    size_t game_ticks;
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
            , board_size(10)
            , board(10, 10, 0) {
        int mid = board_size / 2;
        // Put the snake in the middle
        board[mid][mid] = 1;
        snake = Snake(mid, mid, mid, mid);
    }

    SnakesGame(size_t _board_size)
            : game_ticks(0)
            , board_size(_board_size)
            , board(_board_size, _board_size, 0) {
        int mid = board_size / 2;
        // Put the snake in the middle
        board[mid][mid] = 1;
        snake = Snake(mid, mid, mid, mid);
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
        // collided with wall(1): -50%
        // collided with body(2): -45%
    int run(bool display=false) {
        int game_over_value = -1;
        // display_game() (optional)

        while(game_over_value == -1) {
            // Input
                // only start the game beyond this point
            // Simulate
                // Checks
                    // if collided with wall
                        // game_over_value = 1;
                    // if collided with body
                        // game_over_value = 2;
                // update

            // display_game() (optional)
            // Input (input here because the first input was outside this loop)
            game_ticks++;
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
    int run_net(bool display, SimpleNet& net) {
        int game_over_value = -1;
        
        // display_game() (optional)
        if(display) {
            // Clearscr
            std:: cout << "\033[H;2J";
            display_game();
        }

        while(game_over_value == -1) {
            // Input
            std::vector<float> state = getState();
            std::vector<float> output = net.predict(state);
            
            int move = 0;
            float highest_dist = -999999999999.0f; // I don't know flaot max
            for(int i = 0; i < output.size(); i++) {
                if(output[i] > highest_dist) {
                    move = i + 1;
                    highest_dist = output[i];
                }
            }

            // == Network's move ===== //
            // doing this may make the network to cheat depending on my fitness assessment
            snake.direction = (move == 0) ? move : snake.direction;
            // ======================= //


            // THIS PART AI HELPP
            // Simulate
                // Head move to direction

                // Checks
                    // if collided with wall
                        // game_over_value = 1;
                    // if collided with body
                        // game_over_value = 2;
                // update



            // display_game() (optional)
            if(display) {
                // Clearscr
                std:: cout << "\033[H;2J";
                display_game();
            }
            
            game_ticks++;
        }

        return game_over_value;
    }


    // make this float because my net only accepts float
    // reconsider later
    std::vector<float> getState() {
        size_t state_size = // board
                            board.data.size()
                            // snake.direction
                            + 1 
                            // snake.size
                            + 1
                            // snake.head_x
                            + 1
                            // snake.head_y
                            + 1
                            // snake.tail_x
                            + 1
                            // snake.tail_y
                            + 1
                            // apple.x
                            + 1
                            // apple.y
                            + 1
                            // board_size
                            + 1
                            // game_ticks
                            + 1;

        std::vector<float> state(state_size);

        int i;
        for(i = 0; i < board.data.size(); i++) {
            state[i] = board.data[i];
        }

        state[i++] = float(snake.direction);
        state[i++] = float(snake.size);
        state[i++] = float(snake.head_x);
        state[i++] = float(snake.head_y);
        state[i++] = float(snake.tail_x);
        state[i++] = float(snake.tail_y);
        state[i++] = float(apple.x);
        state[i++] = float(apple.y);
        state[i++] = float(board_size);
        state[i++] = float(game_ticks);
        
        return state;
    }

    // Screenshot only: renders the current game state once.
    void display_game() {
        enableVt();

        // Clear screen and move cursor home
        std::cout << "\x1b[2J\x1b[H";

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

                std::cout << "\x1b[48;2;" << r << ";" << g << ";" << b << "m  \x1b[0m";
            }
            std::cout << "\n";
        }
        std::cout << std::flush;
    }
};