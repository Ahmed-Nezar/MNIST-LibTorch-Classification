#include <torch/torch.h>
#include <iostream>
#include "model.hpp"

using namespace std;

template <typename DataLoader, typename Optimizer, typename Loss>
void train_model(NeuralNetwork &model, int epochs, DataLoader& train_data_loader, DataLoader& valid_data_loader ,Optimizer &optimizer, Loss &criterion, torch::Device device) 
{
    // Training the model
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        model.train();
        size_t batch_index = 0;
        float TRAIN_LOSS = 0;
        float VALID_LOSS = 0;
        for (auto& batch : *train_data_loader) {
            batch.data = batch.data.to(device);
            batch.target = batch.target.to(device);
            optimizer.zero_grad();
            torch::Tensor prediction = model.forward(batch.data);
            torch::Tensor loss = criterion(prediction, batch.target);
            loss.backward();
            optimizer.step();
            batch_index++;
            TRAIN_LOSS += loss.item<float>();
        }
        cout << "Epoch: " << epoch << " | Train Loss: " << TRAIN_LOSS/batch_index; // Average Training loss for the epoch
        batch_index = 0;
        for(auto& batch: *valid_data_loader){
            model.eval();
            batch.data = batch.data.to(device);
            batch.target = batch.target.to(device);
            torch::Tensor prediction = model.forward(batch.data);
            torch::Tensor loss = criterion(prediction, batch.target);
            batch_index++;
            VALID_LOSS += loss.item<float>();
        }
        // Print epoch, loss, accuracy
        cout <<" Valid Loss: " << VALID_LOSS/batch_index <<endl; // Average Validation loss for the epoch

        // Check accuracy
        model.eval();
        int correct = 0;
        int total = 0;
        for (const auto& batch : *train_data_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            auto output = model.forward(data);
            auto prediction = output.argmax(1);
            total += target.size(0);
            correct += prediction.eq(target).sum().item<int>();
        }
        cout << "Train Accuracy: " << (float)correct / total << " ";
        correct = 0;
        total = 0;
        for (const auto& batch : *valid_data_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            auto output = model.forward(data);
            auto prediction = output.argmax(1);
            total += target.size(0);
            correct += prediction.eq(target).sum().item<int>();
        }
        cout << "Valid Accuracy: " << (float)correct / total << endl;
    }
}

void save_model(NeuralNetwork &model, string path){
    torch::save(model.parameters(), path);
}


