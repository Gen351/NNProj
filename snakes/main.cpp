#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <future>
#include <mutex>
#include <atomic>
#include <queue>
#include <functional>
#include <condition_variable>

#include "../nn_engine/simple_net/simple_net.h"
#include "../nn_engine/net_reader/simple_net_reader.h"
#include "./snake_environment.h"
#include "./snake_trainer.h"

void mutate(SimpleNet& net
        , const float mutation_chance
        , const float sigma);

struct RunSnapshot {
    float fitness;
    size_t gen;
    int score;
    std::string screenshot;
};

std::string renderScreenshot(const SnakesGame& game);
int findNextRunVersion();
RunSnapshot makeSnapshot(const SnakesGame& game, const float fitness, const size_t gen);
void save(const std::string& run_dir
        , const SimpleNet& best
        , const SimpleNet& current
        , const RunSnapshot& best_snap
        , const RunSnapshot& curr_snap
        , const size_t board_size);

// Thread pool for parallel evaluation
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    explicit ThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using return_type = decltype(f());
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }
};

int main(int argc, char* argv[]) {
    std::string continue_training = "";
    int max_gen = 100'000;
    int batch_save = 100;
    float mutation_chance = 0.18f;
    float sigma = 0.05f;
    int pop_size = 50;
    int evals_per_net = 5;

    // Read User Args ============== //
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');
        if (eq == std::string::npos) continue;
        
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
            val = val.substr(1, val.size() - 2);
        }
        
        if (key == "CONTINUE_TRAINING") continue_training = val;
        else if (key == "MAX_GEN") max_gen = std::stoi(val);
        else if (key == "BATCH_SAVE") batch_save = std::stoi(val);
        else if (key == "MUTATION_RATE") mutation_chance = std::stof(val);
        else if (key == "SIGMA") sigma = std::stof(val);
        else if (key == "POP_SIZE") pop_size = std::stoi(val);
        else if (key == "EVALS_PER_NET") evals_per_net = std::stoi(val);
    }
    // ============================= //

    // Path resolution for CONTINUE_TRAINING
    if (!continue_training.empty()) {
        if (!std::filesystem::path(continue_training).is_absolute()) {
            continue_training = "trained_networks/" + continue_training;
        }
        if (continue_training.size() < 11 || continue_training.substr(continue_training.size() - 11) != ".simple_net") {
            continue_training += ".simple_net";
        }
    }

    // Train Variables ============= //
    const int board_size = 20;
    
    // Population ES settings
    const int num_elites = pop_size / 5;        // Top networks preserved unchanged
    const int tournament_k = 5;                 // Tournament selection size
    
    float best_fitness = -99999.0f;
    SimpleNet bestNet;
    RunSnapshot best_snapshot;
    // ============================= //
    
    
    // Need to create now for getStateSize();
    SnakesGame game(board_size);
    
    // Network Variables =========== //
    const size_t input = game.getStateSize();
    const size_t output = 4;
    // ============================= //



    // Network Design ============== //
    // The 18 inputs are already hand-crafted sensors (rays, apple dx/dy), so
    // the net only has to combine "go toward apple" with "avoid obstacle":
    // that needs modest width and little depth. The previous 18->64->128->32->4
    // had ~13.5k weights: way too big for mutation-only ES with a population
    // of 20 (children mutate ~40% of weights, so big nets search almost
    // randomly). This one is ~1.2k weights; TANH everywhere (monotonic, so
    // argmax selection is unchanged, and no dead RELU outputs).
    auto make_net = [&](void) -> SimpleNet {
        SimpleNet n;

        n.addLayer(std::make_unique<SimpleLayer>(input, 36));
        n.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::TANH));
        
        n.addLayer(std::make_unique<SimpleLayer>(36, 50));
        n.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::TANH));
        
        n.addLayer(std::make_unique<SimpleLayer>(50, 18));
        n.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::TANH));

        n.addLayer(std::make_unique<SimpleLayer>(18, output));
        n.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::TANH));

        return n;
    };
    
    SimpleNet net;
    if(continue_training.empty()) {
        net = make_net();
    } else {
        try {
            net = SimpleNetReader::load(continue_training);
            
            const SimpleLayer* first = dynamic_cast<const SimpleLayer*>(net.layers[0].get());
            if (!first) throw std::runtime_error("First layer not SimpleLayer");
            if (first->weights.cols() != input) 
                throw std::runtime_error("Input size mismatch: expected " + std::to_string(input) + 
                                         ", got " + std::to_string(first->weights.cols()));
            
            const SimpleLayer* last = nullptr;
            for (auto it = net.layers.rbegin(); it != net.layers.rend(); ++it) {
                last = dynamic_cast<const SimpleLayer*>(it->get());
                if (last) break;
            }
            if (!last || last->weights.rows() != output)
                throw std::runtime_error("Output size mismatch: expected " + std::to_string(output));
            
        } catch(const std::exception& e) {
            throw std::runtime_error("Network Design: Invalid Model Load: " + std::string(e.what()));
        }
    }
    // ============================= //



// Train Loop ================== //
    struct Individual {
        SimpleNet net;
        float fitness = -99999.0f;
        int score = 0;
        SnakesGame best_game;   // final state of this net's best single run
    };
    
    // Initialize population
    std::vector<Individual> population;
    population.reserve(pop_size);
    
    // Seed with loaded net + mutations
    population.push_back({net, -99999.0f, 0});
    for (int i = 1; i < pop_size; ++i) {
        SimpleNet mutant = net;
        mutate(mutant, mutation_chance, sigma);
        population.push_back({std::move(mutant), -99999.0f, 0});
    }
    
    // Pin the run version once per session
    const std::string run_dir = "trained_networks/run_" + std::to_string(findNextRunVersion());
    
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_idx(0, pop_size - 1);
    
    // Thread pool for parallel evaluation
    const int num_threads = std::thread::hardware_concurrency();
    ThreadPool pool(num_threads > 0 ? num_threads / 2 : 4);
    
    auto eval_individual = [&](Individual& ind) -> float {
        float total_fitness = 0.0f;
        int max_score = 0;
        float best_single = -1e30f;
        SnakesGame best_game(board_size);
        for (int e = 0; e < evals_per_net; ++e) {
            SnakesGame local_game(board_size);
            const float f = SnakeTrainer::run_net(local_game, ind.net, false, 0).fitness;
            total_fitness += f;
            if (f > best_single) { best_single = f; best_game = local_game; }
            max_score = std::max(max_score, (int)local_game.snake.size);
        }
        ind.fitness = total_fitness / evals_per_net;
        ind.score = max_score;
        ind.best_game = std::move(best_game);
        return ind.fitness;
    };
    
    auto eval_population = [&](std::vector<Individual>& pop) {
        std::vector<std::future<float>> futures;
        futures.reserve(pop.size());
        
        for (auto& ind : pop) {
            futures.push_back(pool.enqueue([&ind, &eval_individual]() -> float {
                return eval_individual(ind);
            }));
        }
        
        // Wait for all evaluations to complete
        for (auto& fut : futures) fut.get();
    };
    
    // Initial evaluation
    eval_population(population);
    
    // Sort by fitness descending
    auto sort_pop = [&](void) {
        std::sort(population.begin(), population.end(), 
            [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
    };
    sort_pop();
    
    bestNet = population[0].net;
    best_fitness = population[0].fitness;
    best_snapshot = makeSnapshot(population[0].best_game, best_fitness, 0);
    
    int gen_count = 0;
    
    while(max_gen == -1 || gen_count <= max_gen) {
        // Evaluate all in parallel (re-evaluate elites too for noise handling)
        eval_population(population);
        sort_pop();
        
        // Update global best
        if (population[0].fitness > best_fitness) {
            bestNet = population[0].net;
            best_fitness = population[0].fitness;
            best_snapshot = makeSnapshot(population[0].best_game, best_fitness, gen_count);
        }
        
        // Display
        std::cout << "Gen " << gen_count
                  << " | Best: " << best_fitness
                  << " | Pop Best: " << population[0].fitness
                  << " | Pop Avg: ";
        float avg_fit = 0; for (auto& p : population) avg_fit += p.fitness; avg_fit /= pop_size;
        std::cout << avg_fit
                  << " | Best Score: " << best_snapshot.score
                  << " | Pop Best Score: " << population[0].score
                  << "\n";
        
        // Save
        if(gen_count != 0 && gen_count % batch_save == 0) {
            std::cout << "Saving...\n";
            save(run_dir
                , bestNet
                , population[0].net
                , best_snapshot
                , makeSnapshot(population[0].best_game, population[0].fitness, gen_count)
                , board_size);
            std::cout << "Saved: " << run_dir << "\n";
        }
        gen_count++;
        
        // Selection & Reproduction
        std::vector<Individual> next_gen;
        next_gen.reserve(pop_size);
        
        // Elitism: keep top num_elites unchanged
        for (int i = 0; i < num_elites; ++i) {
            next_gen.push_back(population[i]);
        }
        
        // Generate offspring via tournament selection + mutation
        while ((int)next_gen.size() < pop_size) {
            // Tournament selection
            int best_idx = dist_idx(rng);
            for (int t = 1; t < tournament_k; ++t) {
                int idx = dist_idx(rng);
                if (population[idx].fitness > population[best_idx].fitness) best_idx = idx;
            }
            
            SimpleNet child = population[best_idx].net;
            mutate(child, mutation_chance, sigma);
            next_gen.push_back({std::move(child), -99999.0f, 0});
        }
        
        population = std::move(next_gen);
    }
    
    // ============================= //




    return 0;
}



static bool shouldMutate(std::mt19937& engine, const float chance) {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(engine) < chance;
}

static float gauss(std::mt19937& engine, const float sigma) {
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(engine) * sigma;
}

void mutate(SimpleNet& net
        , const float mutation_chance
        , const float sigma) {
    static thread_local std::mt19937 engine(std::random_device{}());

    for(auto& l : net.layers) {

        // wtf is this, is this because of my bad implementaion?
        SimpleLayer* layer = dynamic_cast<SimpleLayer*>(l.get());
        if(!layer) continue; // activation layer        

        for(float& w : layer->weights.data)
            if(shouldMutate(engine, mutation_chance)) w += gauss(engine, sigma);
        for(float& b : layer->biases)
            if(shouldMutate(engine, mutation_chance)) b += gauss(engine, sigma);
    }
}


std::string renderScreenshot(const SnakesGame& game) {
    std::string out;
    const size_t n = game.board_size + 2;
    for(size_t row = 0; row < n; row++) {
        for(size_t col = 0; col < n; col++) {
            if(row == 0 || col == 0 || row == n - 1 || col == n - 1) {
                out += (row == 0 || row == n - 1) ? '-' : '|';
            }
            else if(col - 1 == (size_t)game.snake.head_x && row - 1 == (size_t)game.snake.head_y) {
                out += 'S';
            }
            else if(game.apple.x != -1
            && col - 1 == (size_t)game.apple.x
            && row - 1 == (size_t)game.apple.y) {
                out += 'A';
            }
            else if(game.board(row - 1, col - 1) == 1) {
                out += 's';
            }
            else {
                out += ' ';
            }
        }
        out += "\n";
    }
    return out;
}


int findNextRunVersion() {
    int max_ver = 0;
    if(!std::filesystem::exists("trained_networks")) return 1;

    for(const std::filesystem::directory_entry& entry
            : std::filesystem::directory_iterator("trained_networks")) {
        const std::string name = entry.path().filename().string();
        if(name.rfind("run_", 0) != 0) continue;
        try {
            const int ver = std::stoi(name.substr(4));
            if(ver > max_ver) max_ver = ver;
        } catch(...) {
            // Not a run_<N> folder, skip it
        }
    }
    return max_ver + 1;
}


RunSnapshot makeSnapshot(const SnakesGame& game, const float fitness, const size_t gen) {
    return RunSnapshot{fitness, gen, (int)game.snake.size, renderScreenshot(game)};
}


void save(const std::string& run_dir
        , const SimpleNet& best
        , const SimpleNet& current
        , const RunSnapshot& best_snap
        , const RunSnapshot& curr_snap
        , const size_t board_size) {
    std::filesystem::create_directories(run_dir);

    // Atomic: write to a temp file, then rename over the target.
    // A Ctrl+C mid-save can no longer leave a half-written .simple_net behind.
    // Retries against transient locks (antivirus/editor handles);
    // on persistent failure keep the tmp file and warn, never kill training.
    auto replaceFile = [&](const std::string& tmp, const std::string& target) {
        for(int attempt = 0; attempt < 10; attempt++) {
            std::error_code ec;
            std::filesystem::rename(tmp, target, ec);
            if(!ec) return;
            if(ec != std::errc::permission_denied) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::cerr << "save: WARNING: could not replace " << target
                  << " (kept " << tmp << "); next save will retry\n";
    };

    SimpleNetReader::save(run_dir + "/best.tmp", best);
    replaceFile(run_dir + "/best.tmp.simple_net", run_dir + "/best.simple_net");
    SimpleNetReader::save(run_dir + "/current.tmp", current);
    replaceFile(run_dir + "/current.tmp.simple_net", run_dir + "/current.simple_net");

    std::ofstream perf(run_dir + "/performance.txt");
    if(!perf.is_open())
        throw std::runtime_error("save: failed to open " + run_dir + "/performance.txt");

    perf << "BOARD_SIZE: " << board_size << "\n";
    perf << "Best;\n";
    perf << "    Fitness         : " << best_snap.fitness << "\n";
    perf << "    Gen             : " << best_snap.gen << "\n";
    perf << "    Score           : " << best_snap.score << "\n";
    perf << "Game Over Screenshot:\n";
    perf << best_snap.screenshot;
    perf << "Current;\n";
    perf << "    Fitness         : " << curr_snap.fitness << "\n";
    perf << "    Gen             : " << curr_snap.gen << "\n";
    perf << "    Score           : " << curr_snap.score << "\n";
    perf << "Game Over Screenshot:\n";
    perf << curr_snap.screenshot;

    if(!perf)
        throw std::runtime_error("save: failed writing " + run_dir + "/performance.txt");
}