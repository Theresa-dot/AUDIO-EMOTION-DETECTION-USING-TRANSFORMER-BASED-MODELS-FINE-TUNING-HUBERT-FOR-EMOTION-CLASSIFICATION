# AUDIO-EMOTION-DETECTION-USING-TRANSFORMER-BASED-MODELS-FINE-TUNING-HUBERT-FOR-EMOTION-CLASSIFICATION

---

# Audio Emotion Detection Using Transformer-Based Models

This project leverages **HuBERT (Hidden-Unit BERT)** for accurate emotion classification from audio signals. The system focuses on offline processing of pre-recorded audio for high accuracy, making it suitable for applications like customer service, mental health monitoring, and interactive AI.

---

## 🚀 Features

- **Transformer-Based Emotion Detection**: Fine-tuned HuBERT model for audio emotion classification.
- **Advanced Preprocessing**: Noise reduction, feature extraction, and signal normalization using Librosa.
- **Data Augmentation**: Robust training with techniques like pitch shifting, time-stretching, and noise addition.
- **Real-Time Experiment Tracking**: Visualize metrics like accuracy and loss using Weights & Biases (W&B).
- **Offline Processing**: Designed for reliable emotion detection in pre-recorded audio.
- **Wide Application Scope**: Suitable for customer service sentiment analysis, mental health monitoring, and empathetic AI systems.

---

## 📊 Dataset: CREMA-D

| **Feature**              | **Details**                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Name**                 | Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)               |
| **Size**                 | 7,442 audio clips                                                         |
| **Emotions**             | Anger, Disgust, Fear, Happiness, Neutral, Sadness                         |
| **Diversity**            | Features recordings from 91 actors with varying accents and speech styles |
| **Usage**                | Preprocessed using tools like Librosa for optimal feature extraction      |

---

## 📈 Results

- **Accuracy**: Achieved **77.21%** test accuracy on unseen data.
- **ROC-AUC**: Scored **0.771**, reflecting strong classification performance.
- **Error Analysis**: Highlighted confusion in closely related emotions, e.g., Neutral vs. Calm.

---

## 🛠️ Technologies Used

### Libraries and Frameworks
- **[Transformers](https://huggingface.co/transformers/)**: Fine-tuning HuBERT and model management.
- **[Librosa](https://librosa.org/)**: Audio preprocessing and feature extraction.
- **[PyTorch](https://pytorch.org/)**: Core framework for model training.
- **[Weights & Biases](https://wandb.ai/)**: Real-time training visualization and monitoring.

### Tools
- **HuBERT**: Pre-trained transformer model for self-supervised speech representation learning.
- **Jupyter Notebook**: For interactive code development and debugging.

---

## ⚙️ Setup

### Prerequisites
- Python 3.7 or higher.
- GPU-enabled machine (recommended for faster training).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/audio-emotion-detection.git
   cd audio-emotion-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation
1. Download the CREMA-D dataset.
2. Place the dataset in the `data/CREMA-D` directory.
3. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```

---

## ▶️ Usage

### Training the Model
1. Configure parameters in `config.py`.
2. Train the model:
   ```bash
   python train.py
   ```

### Evaluating the Model
Run the evaluation script to test model performance:
   ```bash
   python evaluate.py
   ```

### Real-Time Inference (Optional)
Perform real-time emotion classification:
   ```bash
   python infer.py --audio_file path/to/audio.wav
   ```
   The script outputs the predicted emotion.

---

## 🌐 Future Directions

- Add real-time emotion detection for interactive applications.
- Extend datasets to include accents, noisy environments, and subtle emotional states.
- Explore multi-modal emotion recognition combining audio, text, and visual data.

---
