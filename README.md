# Pneumonia Detection App

AI-powered chest X-ray analysis system using ResNet152 for pneumonia detection.

## Features
- Deep learning model (ResNet152V2) trained on chest X-rays
- React frontend with modern UI
- Flask backend for predictions
- Dockerized deployment

## Tech Stack
- **Frontend:** React, Tailwind CSS
- **Backend:** Python, Flask, TensorFlow
- **Model:** ResNet152V2 (Transfer Learning)
- **Deployment:** Docker, Docker Compose

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed
- Trained model file: `resnet152_pneumonia_baseline.keras`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-detection-app.git
cd pneumonia-detection-app
```

2. Add your trained model:
   - Place `resnet152_pneumonia_baseline.keras` in the `backend/` folder

3. Build and run with Docker:
```bash
docker-compose up --build
```

4. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## Usage

1. Open http://localhost:3000 in your browser
2. Upload a chest X-ray image (PNG/JPG)
3. Click "Analyze X-Ray"
4. View the prediction results

## Model Training

The model was trained using:
- Dataset: Chest X-Ray Images (Pneumonia) from Kaggle
- Architecture: ResNet152V2 (transfer learning)
- Framework: TensorFlow/Keras
- Training details: See training notebook in repository

## Important Notes

⚠️ **The trained model file is NOT included in this repository** due to its large size. 

To use this project:
1. Train your own model using the provided training code
2. Download the model file
3. Place it in the `backend/` folder
4. Run docker-compose