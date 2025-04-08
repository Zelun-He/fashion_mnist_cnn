
👟 Fashion-CNN: Image Classification with CNN & ResNet
This is a simple yet powerful implementation of two Convolutional Neural Network (CNN) architectures — a custom BaseNet and a ResNet-style model — for classifying clothing images from the Fashion-MNIST dataset.

🧠 Key Features
Two CNN architectures: BaseNet and ResNet-style

Model evaluation using accuracy and loss metrics

Data augmentation support to improve generalization

Built with PyTorch in a single-file script for simplicity

🚀 Getting Started
Requirements
Python 3.7+

PyTorch

torchvision

matplotlib

Installation
bash
Copy
pip install torch torchvision matplotlib
🏋️ Usage
Run the script and specify the model you want to train:

bash
Copy
python fashion_cnn.py --model basenet
python fashion_cnn.py --model resnet
Optional arguments:

--epochs: Number of training epochs (default: 10)

--batch-size: Batch size for training (default: 64)

📊 Results
Model	Accuracy
BaseNet	~XX%
ResNet	~YY%
ResNet generally performs better due to its deeper architecture and residual connections.

📚 Dataset
Fashion-MNIST is a dataset of 70,000 grayscale images across 10 fashion categories such as T-shirts, trousers, and sneakers.

📌 Future Improvements
Add more CNN variations (e.g., VGG or MobileNet)

Experiment with different optimizers and schedulers

Save and load trained models
