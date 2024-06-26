#include "utills.hpp"

using namespace std;

int main() {

    // Define Cuda device
    torch::Device device(torch::kCUDA);
    torch::manual_seed(42);

    NeuralNetwork model(784, 10);
    model.to(device);
    
    // Defining Adam Optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    // Defining Cross Entropy Loss
    torch::nn::CrossEntropyLoss criterion;

    // Loading MNIST dataset training data
    auto dataset = torch::data::datasets::MNIST("./MNIST")
        .map(torch::data::transforms::Stack<>());

    auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset), 64);

    auto valid_dataset = torch::data::datasets::MNIST("./MNIST", torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Stack<>());
    auto valid_dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(valid_dataset), 64);

    int epochs = 10;
    train_model(model, epochs, train_data_loader,valid_dataloader,optimizer, criterion, device);
    
    save_model(model, "model.pth");

    return 0;
}
