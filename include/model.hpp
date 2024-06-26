#include <torch/torch.h>


struct NeuralNetwork: torch::nn::Module{
    NeuralNetwork(int input_size, int output_size) :

        fc1(input_size, 64),
        fc2(64, 32),
        fc3(32,output_size)
        {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
            register_module("fc3", fc3);
        }

        torch::Tensor forward(torch::Tensor x){
            x = x.view({x.size(0), -1});
            x = torch::relu(fc1->forward(x));
            x = torch::relu(fc2->forward(x));
            x = torch::softmax(fc3->forward(x), 1);
            return x;
        }

        torch::nn::Linear fc1, fc2, fc3;

};
