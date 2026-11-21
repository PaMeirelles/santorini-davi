#include <iostream>
#include "evaluation.h"
int main(int argc, char** argv) {
    std::cerr << "start\n";
    try {
        std::cerr << "before cpu tensor\n";
        auto t = torch::zeros({1,35,5,5}, torch::kFloat32);
        std::cerr << "before cuda tensor\n";
        auto cuda = t.to(torch::kCUDA);
        std::cerr << "before model load\n";
        auto model = torch::jit::load(argv[1]);
        std::cerr << "before model cuda\n";
        model.to(torch::kCUDA); model.eval();
        std::cerr << "before forward\n";
        torch::NoGradGuard guard;
        auto result = model.forward({cuda}).toTuple();
        std::cerr << "after forward " << result->elements()[1].toTensor() << "\n";
        Santorini::Board board(argv[2]);
        std::cerr << "before evaluator\n";
        auto nn = Santorini::evaluate_board_nn(board, 0);
        std::cerr << "after evaluator " << nn.value << "\n";
    } catch(const std::exception& e) {std::cerr << "exception " << e.what() << "\n"; return 1;}
}
