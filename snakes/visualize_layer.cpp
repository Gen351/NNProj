#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <stdexcept>

#include "../nn_engine/simple_net/simple_net.h"
#include "../nn_engine/simple_net/simple_layer.h"
#include "../nn_engine/net_reader/simple_net_reader.h"
#include "./snake_environment.h"
#include "./snake_trainer.h"


#ifdef _WIN32
#include <windows.h>
void enableAnsiSupport() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &dwMode)) {
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }
}
#else
void enableAnsiSupport() {} // No-op for Linux/macOS
#endif


std::string get_pixel(const std::string& act_type, float val, float min, float max, const std::string& text);

void print_layer_2d(Matrix<float> layer, bool show_val);
void show_net_2d(std::vector<float> input, const SimpleNet& net, bool show_val);

void print_layer(std::vector<float> layer, const std::string& act_type, bool first_layer, bool show_val);
void show_net(std::vector<float> input, const SimpleNet& net);
SnakesGame load_board(const std::string& perf_path);

int main(int argc, char* argv[]) {
	enableAnsiSupport();

    srand(std::time(nullptr));

    if(argc < 2) {
        std::cout << "Usage: test <model.simple_net> [speed_ms] [board_size] [performance.txt]\n"
                  << "  model.simple_net : path to a trained net (e.g. ./trained_networks/run_19/best.simple_net)\n"
                  << "  speed_ms         : ms per frame (default 500)\n"
                  << "  board_size       : board size (ignored when performance.txt is given)\n"
                  << "  performance.txt  : optional run_<N>/performance.txt; loads the 'Best' game-over\n"
                  << "                     screenshot state instead of starting a fresh game\n"
                  << "  show_val         : optional show the weights on the net\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const int speed_ms = (argc >= 3) ? std::atoi(argv[2]) : 500;
    const int suggested_board_size = (argc >= 4) ? std::atoi(argv[3]) : 35;
    const std::string perf_path = (argc >= 5) ? argv[4] : "";
    const bool show_val = (argc >= 6) ? (std::string(argv[5]) == "true" || std::string(argv[5]) == "1") : false; 

    SimpleNet net;
    
	try {net = SimpleNetReader::load(model_path);}
	catch(const std::exception& e) { std::cerr << "test: failed to load model: " << e.what() << "\n"; return 1; }

	if(net.layers.empty()) { std::cerr << "test: model has no layers: " << model_path << "\n"; return 1; }

    const SimpleLayer* first = dynamic_cast<const SimpleLayer*>(net.layers[0].get());
    if(!first) { std::cerr << "test: first layer is not a SimpleLayer\n"; return 1; }
    


    if(!perf_path.empty()) {
        SnakesGame game;
        try { game = load_board(perf_path); }
        catch(const std::exception& e) { std::cerr << "visualize_layer: failed to load state: " << e.what() << "\n"; return 1; }

        std::cout << "Loaded state from " << perf_path
                  << " | board: " << game.board_size << "x" << game.board_size
                  << " | snake: " << game.snake.size
                  << " | apple: " << (game.apple.x == -1 ? "none" : "(" + std::to_string(game.apple.x) + ", " + std::to_string(game.apple.y) + ")")
                  << "\n";

                  
        const std::vector<float> state = game.getState(); 

        const std::vector<float> output = net.predict(state);
        const int move = SnakeTrainer::pickMove(game, output);
        game.snake.direction = move;

        // PRINT ==================================== //
        game.display_game(true);
        show_net_2d(state, net, show_val);
        // ========================================== //

        std::cout << "> Net picks move    :" << move << " (1=turn left, 2=straight, 3=turn right)\n";
        std::cout << "> Snapshot          : " << perf_path << "\n";
        std::cout << "> Model             : " << model_path << "\n";
    } else {
        std::cout << "Usage: test <model.simple_net> [speed_ms] [board_size] [performance.txt]\n"
                    << "  model.simple_net : path to a trained net (e.g. ./trained_networks/run_19/best.simple_net)\n"
                    << "  speed_ms         : ms per frame (default 500)\n"
                    << "  board_size       : board size (ignored when performance.txt is given)\n"
                    << "  performance.txt  : optional run_<N>/performance.txt; loads the 'Best' game-over\n"
                    << "                     screenshot state instead of starting a fresh game\n"
                    << "  show_val         : optional show the weights on the net\n";
        return 1;
    }

	return 1;
};



void show_net(std::vector<float> input, const SimpleNet& net) {
	std::cout << "\n\n";

	std::vector<float> output = input;
	print_layer(output, "SIGM", true, true);
    std::cout << "\n";

	for(const auto& l : net.layers) {
		output = l->forward(output);

		if(l->getType() == "ACT") {
			// Safely cast the base pointer to a SimpleActivationLayer pointer
			const auto* act_layer = dynamic_cast<const SimpleActivationLayer*>(l.get());
			
            const std::string act_type = act_layer->getActType(); // "SIGM", "TANH", etc.
            print_layer(output, act_type, false, true);	
            std::cout << "\n";
		}
	}
}

void show_net_2d(std::vector<float> input, const SimpleNet& net, bool show_val) {
	std::cout << "\n\n";

	std::vector<float> output = input;
	print_layer(output, "SIGM", true, show_val);
    std::cout << "\n\n\n";

	for(const auto& l : net.layers) {
		output = l->forward(output);

		if(l->getType() == "ACT") {
			// Safely cast the base pointer to a SimpleActivationLayer pointer
			const auto* act_layer = dynamic_cast<const SimpleActivationLayer*>(l.get());			
            const std::string act_type = act_layer->getActType(); // "SIGM", "TANH", etc.
            
            print_layer(output, act_type, false, show_val);	
            std::cout << "\n\n\n";
		} else {
            const auto* simple_layer = dynamic_cast<const SimpleLayer*>(l.get());
            
            print_layer_2d(simple_layer->weights, show_val);
            print_layer(simple_layer->biases, "LEAK", false, show_val);
            std::cout << "\n";
        }
	}
}

void print_layer_2d(Matrix<float> layer, bool show_val) {    
    // Get min, max
    float min = layer.data[0];
    float max = layer.data[0];

    for(int i = 1; i < layer.data.size(); i++) {
        float temp = layer.data[i];
        if(temp > max) max = temp;
        if(temp < min) min = temp;
    }

    // create the buffer
    std::string vis_layer = "\t";

    for(int i = 0; i < layer.cols(); i++) {
        for(int j = 0; j < layer.rows(); j++) {
            float val = layer(i, j);
            if(show_val) {
                std::ostringstream stream;
                stream << std::fixed << std::setprecision(3) << val;
                vis_layer += get_pixel("LEAK", val, min, max, stream.str());
            } else {
                vis_layer += get_pixel("LEAK", val, min, max, "  ");
            }
            vis_layer += " ";
        }
        vis_layer += "\n\t";
    }

	std::cout << vis_layer << "\n" << std::flush;
}

void print_layer(std::vector<float> layer, const std::string& act_type, bool first_layer, bool show_val) {
	float min, max;
	if(act_type == "RELU" || act_type == "LEAK") {
		min = layer[0];
		max = layer[0];
		for(int i = 1; i < layer.size(); i++) {
			if(layer[i] > max) max = layer[i];
			if(layer[i] < min) min = layer[i];
		}
	} else { // default to sigmoid
		min = 0.0f;
		max = 1.0f;
	}

	std::string vis_layer = "\t";
    if(first_layer) {
        vis_layer += "         W A L L                   B O D Y                  A P P L E\n\t";
    }

    int i = 0;
	for(const float& val : layer) {
        if(first_layer && i % 8 == 0 && i != 0) {
            vis_layer += "~~ ";
        }
        i++;

        if(show_val) {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(3) << val;
            
            vis_layer += get_pixel(act_type, val, min, max, stream.str());
        } else {
            vis_layer += get_pixel(act_type, val, min, max, "  ");
        }
        vis_layer += " ";
	}

	std::cout << vis_layer << "\n" << std::flush;
}



std::string get_pixel(const std::string& act_type, float val, float min, float max, const std::string& text) {
    // pick min, max
	if (act_type == "SIGM") {
        // Sigmoid is natively [0, 1]
		min = 0.0f;
		max = 1.0f;
    } else if (act_type == "TANH") {
		min = -1.0f;
		max = 1.0f;
	} else {
        // keep the passed min, max
		// calculated preemptively and passed on the arg
    }

	// clamp the value to the bounds
    val = std::max(min, std::min(max, val));

	// normalize value from [min, max] down to [0, 1]
    float t = 0.5f;
    if (max > min) {
        t = (val - min) / (max - min);
    }

    int r = 0, g = 0, b = 0;
    
    // interpolate color using the normalized [0, 1] range (t)
    if (t <= 0.5f) {
        float t_prime = t * 2.0f; // Scale first half to [0, 1]
        r = 0;
        g = static_cast<int>(t_prime * 255.0f);
        b = static_cast<int>((1.0f - t_prime) * 255.0f);
    } else {
        float t_prime = (t - 0.5f) * 2.0f; // Scale second half to [0, 1]
        r = static_cast<int>(t_prime * 255.0f);
        g = 255;
        b = 0;
    }	

	std::string rgb_str = std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b);
    std::string text_color = std::to_string(255 - r) + ";" + std::to_string(255 - g) + ";" + std::to_string(255 - b);
    
    // 38;2 = Foreground RGB, 48;2 = Background RGB
    return "\033[38;2;" + text_color + ";48;2;" + rgb_str + "m" 
           + text + "\033[0m";
}






SnakesGame load_board(const std::string& perf_path) {
    std::ifstream file(perf_path);
    if(!file.is_open())
        throw std::runtime_error("load_board: cannot open " + perf_path);

    int board_size = 0;
    std::vector<std::string> lines;
    {
        std::string line;
        while(std::getline(file, line)) {
            if(!line.empty() && line.back() == '\r') line.pop_back();
            lines.push_back(line);
        }
    }

    // Parse "BOARD_SIZE: <N>".
    bool have_size = false;
    for(const auto& line : lines) {
        const std::string prefix = "BOARD_SIZE:";
        if(line.compare(0, prefix.size(), prefix) == 0) {
            try { board_size = std::stoi(line.substr(prefix.size())); }
            catch(...) { board_size = -1; }
            have_size = true;
            break;
        }
    }
    if(!have_size || board_size <= 0)
        throw std::runtime_error("load_board: missing/invalid BOARD_SIZE in " + perf_path);

    // Locate the "Best;" block's "Game Over Screenshot:" marker (always Best).
    size_t marker = lines.size();
    for(size_t i = 0; i < lines.size(); ++i) {
        if(lines[i].rfind("Best;", 0) == 0) {
            for(size_t j = i + 1; j < lines.size(); ++j) {
                if(lines[j].find("Game Over Screenshot:") != std::string::npos) {
                    marker = j;
                    break;
                }
            }
            break;
        }
    }
    if(marker == lines.size())
        throw std::runtime_error("load_board: no 'Best' screenshot found in " + perf_path);

    // The next board_size + 2 rows are the screenshot.
    const size_t n = (size_t)board_size + 2;
    if(marker + n >= lines.size())
        throw std::runtime_error("load_board: screenshot truncated in " + perf_path);

    SnakesGame::Pos head{ -1, -1 };
    SnakesGame::Pos apple{ -1, -1 };
    std::vector<SnakesGame::Pos> body;
    bool have_head = false, have_apple = false;

    for(size_t r = 0; r < n; ++r) {
        const std::string& row = lines[marker + 1 + r];
        if(row.size() < n)
            throw std::runtime_error("load_board: screenshot row too short in " + perf_path);

        const bool border = (r == 0 || r == n - 1);
        for(size_t c = 0; c < n; ++c) {
            const char ch = row[c];
            if(border) continue; // dash border row
            if(c == 0 || c == n - 1) {
                if(ch != '|')
                    throw std::runtime_error("load_board: expected '|' border in " + perf_path);
                continue;
            }

            const int x = (int)(c - 1);
            const int y = (int)(r - 1);
            if(ch == 'S') {
                if(have_head)
                    throw std::runtime_error("load_board: multiple heads in " + perf_path);
                head = SnakesGame::Pos{ x, y };
                have_head = true;
            }
            else if(ch == 's') {
                body.push_back(SnakesGame::Pos{ x, y });
            }
            else if(ch == 'A') {
                if(have_apple)
                    throw std::runtime_error("load_board: multiple apples in " + perf_path);
                apple = SnakesGame::Pos{ x, y };
                have_apple = true;
            }
            else if(ch != ' ') {
                throw std::runtime_error(std::string("load_board: unexpected char '") + ch + "' in " + perf_path);
            }
        }
    }

    if(!have_head)
        throw std::runtime_error("load_board: no head 'S' found in " + perf_path);

    // Front = head, then body cells in scan order. The screenshot stores no
    // body ordering, so order is arbitrary; direction stays default (0), which
    // getState() treats as facing up. We only analyze this state, never step().
    std::deque<SnakesGame::Pos> deque;
    deque.push_front(head);
    for(const auto& p : body) deque.push_back(p);

    SnakesGame game(board_size);
    game.setBody(deque);
    game.apple.x = have_apple ? apple.x : -1;
    game.apple.y = have_apple ? apple.y : -1;
    game.snake.direction = 0;
    game.snake.prev_direction = 0;

    return game;
}