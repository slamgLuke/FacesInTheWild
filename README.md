# 📷 CNN for Comparison of Human Faces

## 🚀 Introduction
Welcome to our project on **Human faces comparison**! In this project, we analyze a dataset of images depicting various human faces. Our goal is to use a Siamese neural network for comparison, extract features using convolutional neural networks (CNNs), and make predictions based on the similarity of faces.

## 👥 Authors
- **Carranza Lucas** (202210025)
- **Lazo Kalos** (201510170)
- **Herencia David** (202010399)
- **Chavez Lenin** (202210090)

## 📄 Abstract
We conducted an exploratory study on a dataset of human faces to compare and classify them based on their features. We used a Siamese neural network to extract features and applied pooling to reduce dimensionality. We then used classification metrics to evaluate our model's performance.

## 🔑 Keywords
- **Siamese neural network**
- **Convolutional neural network**
- **Pooling**
- **F1-score**

## 📚 Dataset Overview
We used a subset of the **Human Action Recognition** dataset due to practical and computational constraints. Our dataset includes:
- **Training**: 1760 videos
- **Validation**: 440 videos
- **Testing**: 1000 videos

### 🎬 Labels
- Same
- Diff

## 🛠️ Feature Extraction
We utilized the CNN block to extract features using AlexNet, ResNet-50, ResNet-152, and FaceNet (based on the Inception model) pretrained due to their superior accuracy and the hardware constraints preventing us from retraining FaceNet. 

## 🧠 Siamese Network Architecture
The Siamese network consists of:
- **Convolutional Base**: We experimented with AlexNet, ResNet-50, ResNet-152, and FaceNet for feature extraction.
- **Fully Connected Layer (MLP)**: A multi-layer perceptron (MLP) with a sigmoid activation function at the output layer to produce binary predictions (0 or 1).

## 📊 Loss Functions
We used the contrastive loss function to train our Siamese network, optimizing it to minimize the distance between similar pairs and maximize the distance between dissimilar pairs.

## 🔍 Methodology
We conducted a series of experiments to find the best combination of dataset preprocessing, CNN architecture, and network hyperparameters. The optimization was based on metrics like accuracy, precision, recall, and F1-score.

## 💻 Implementation
Our implementation is available in the [GitHub repository](https://github.com/Hyp3Boy/Clustering_project/tree/main). Check out the **models** folder for the code.

## 📈 Experimentation
We experimented with different configurations to find the best parameters for clustering. Below are some key results:

### Accuracy Scores
| Model       | Accuracy |
|-------------|----------|
| AlexNet     | 0.8424   |
| ResNet-50   | 0.8388   |
| ResNet-152  | 0.8512   |
| FaceNet     | 0.8675   |

### F1-Scores
| Model       | F1-Score |
|-------------|----------|
| AlexNet     | 0.8321   |
| ResNet-50   | 0.8285   |
| ResNet-152  | 0.8390   |
| FaceNet     | 0.8544   |

## 📝 Conclusions
1. **Best Accuracy**: FaceNet performed best, indicating better alignment with the true structure of the data.
2. **Best F1-Score**: FaceNet excelled in balancing precision and recall.
3. **Model Choices**: Each model has its strengths:
    - **AlexNet**: Simpler, faster to train, but less accurate.
    - **ResNet-50/152**: Better accuracy due to deeper architecture.
    - **FaceNet**: Best for face recognition tasks.

## 📚 Future Work
- **Improve Class Distribution**: More balanced data for better clustering.
- **Evaluate Other Models**: Experiment with other advanced models and architectures for better performance.

## 📜 References
- [Siamese Neural Networks for One-Shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Deep Face Recognition with FaceNet](https://doi.org/10.1109/CVPR.2015.7298682)
- [ResNet: Deep Residual Learning for Image Recognition](https://doi.org/10.1109/cvpr.2016.90)
- [AlexNet](https://doi.org/10.48550/arXiv.1404.5997)

Feel free to explore our project and contribute!
