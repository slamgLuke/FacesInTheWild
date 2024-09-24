# 📷 CNN for Comparison of Human Faces

## 🚀 Introduction
Welcome to our project on **Human faces comparison**! In this project, we analyze a dataset of images depicting various human faces. Our goal is to use Siamese and Triplet neural networks for comparison, extract features using convolutional neural networks (CNNs), and make predictions based on the similarity of faces.

## 👥 Authors
- **Carranza Lucas** (202210073)
- **Lazo Kalos** (202210184)
- **Herencia David** (202210408)
- **Chavez Lenin** (202210090)

## 📄 Abstract
We conducted an exploratory study on a dataset of human faces to compare and classify them based on their features. We used Siamese and Triplet neural networks to extract features and applied pooling to reduce dimensionality. We then used classification metrics to evaluate our model's performance.

## 🔑 Keywords
- **Siamese neural network**
- **Triplet neural network**
- **Convolutional neural network**
- **Pooling**
- **F1-score**

## 📚 Dataset Overview
We used a subset of the **Human Action Recognition** dataset due to practical and computational constraints. Our dataset includes:
- **Training**: 1760 videos
- **Validation**: 440 videos
- **Testing**: 1000 videos

### 📷! Labels
- Same (1)
- Diff (0)

## 🛠️ Feature Extraction
We utilized the CNN block to extract features using AlexNet, ResNet-50, ResNet-152 pretraineds on Imagenet1k_v1, and FaceNet (based on the Inception model) pretrained on VGGFace2 due to their superior accuracy and the hardware constraints preventing us from retraining the majority of FaceNet.

## 🧠 Network Architectures

### Siamese Network
The Siamese network consists of:
- **Convolutional Base**: We experimented with AlexNet, ResNet-50, ResNet-152, and FaceNet for feature extraction.
- **Fully Connected Layer (MLP)**: A multi-layer perceptron (MLP) with a sigmoid activation function at the output layer to produce binary predictions (0 or 1).

### Triplet Network
The Triplet network consists of:
- **Convolutional Base**: We used AlexNet for feature extraction.
- **Triplet Loss Function**: We used the triplet loss function to train our network, optimizing it to ensure that the distance between the anchor and positive examples is less than the distance between the anchor and negative examples.

## 📊 Loss Functions
For the Siamese network, we used the binary cross entropy loss function, optimizing it to minimize the distance between similar pairs and maximize the distance between dissimilar pairs. For the Triplet network, we used the triplet loss function.

## 🔍 Methodology
We conducted a series of experiments to find the best combination of dataset preprocessing, CNN architecture, and network hyperparameters. The optimization was based on metrics like accuracy, precision, recall, and F1-score.

## 💻 Implementation
Our implementation is available in the [GitHub repository](https://github.com/slamgLuke/FacesInTheWild/tree/main). Check out the **jupyter notebooks** for the code.

## 📈 Experimentation
We experimented with different architectures to improve the comparison. Below are some key results:

### Siamese Network

#### Accuracy Scores
| Model       | Accuracy |
|-------------|----------|
| AlexNet     | 0.60     |
| ResNet-50   | 0.58     |
| ResNet-152  | 0.69     |
| FaceNet     | 0.97     |

#### F1-Scores
| Model       | F1-Score |
|-------------|----------|
| AlexNet     | 0.60     |
| ResNet-50   | 0.58     |
| ResNet-152  | 0.69     |
| FaceNet     | 0.99     |

### Triplet Network

#### Accuracy Scores
| Model       | Accuracy |
|-------------|----------|
| TripletNN   | 0.74     |

#### F1-Scores
| Model       | F1-Score |
|-------------|----------|
| TripletNN   | 0.81     |

## 📝 Conclusions
1. **Best Accuracy**: FaceNet performed best, indicating better alignment with the true structure of the data.
2. **Best F1-Score**: FaceNet excelled in balancing precision and recall.
3. **Model Choices**: Each model has its strengths:
    - **AlexNet**: Simpler, faster to train, but less accurate.
    - **ResNet-50/152**: Better accuracy due to deeper architecture.
    - **FaceNet**: Best for face recognition tasks.
    - **TripletNN**: Effective in scenarios requiring differentiation between multiple classes.
4. **Data Augmentation and Dataset Expansion:**
   - Given that the dataset used in this project is relatively small (2200 image pairs), it was crucial to implement Data Augmentation techniques to prevent overfitting. Increasing the dataset size by creating modified versions of the images helped the model generalize better.

## 📚 Future Work
1. **Evaluation of Alternative Architectures:**
   - Consider applying other architectures for feature extraction in our Triplet Neural Networks, which have the potential to improve accuracy without significantly increasing the complexity of the model.

2. **Transfer Learning:**
   - Implement and experiment with Transfer Learning techniques, especially in pretrained models with similar accuracy percentages to FaceNet, which have proven effective in facial recognition tasks. This may include retraining the final layers of the model to better adapt to the new dataset's characteristics.

## 📜 References
- [Siamese Neural Networks for One-Shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Deep Face Recognition with FaceNet](https://doi.org/10.1109/CVPR.2015.7298682)
- [ResNet: Deep Residual Learning for Image Recognition](https://doi.org/10.1109/cvpr.2016.90)
- [AlexNet](https://doi.org/10.48550/arXiv.1404.5997)

Feel free to explore our project and contribute!
