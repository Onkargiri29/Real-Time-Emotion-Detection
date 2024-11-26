
# Real-Time Emotion Detection

A Python-based application for detecting human emotions in real-time using a pre-trained Convolutional Neural Network (CNN) model. This project leverages OpenCV for facial detection, TensorFlow/Keras for deep learning, and Flask for web integration.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Scope](#future-scope)
- [Contributors](#contributors)
- [License](#license)

---

## Features
- Real-time emotion detection using facial expressions.
- Supports multiple emotions such as Happy, Sad, Angry, Neutral, and more.
- Web-based interface using Flask for easy interaction.
- Trained on the FER-2013 dataset for robust performance.

---

## Technologies Used
- **Python**: Programming language.
- **TensorFlow/Keras**: Deep learning framework for training and using the emotion detection model.
- **OpenCV**: For face detection and real-time video processing.
- **Flask**: Lightweight web framework for creating the web interface.
- **HTML/CSS/JS**: Frontend for the web application.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Onkargiri29/Real-Time-Emotion-Detection.git
   cd Real-Time-Emotion-Detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # For Linux/Mac
   env\Scripts\activate      # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model (if applicable):
   Ensure the file `emotiondetector.h5` is in the project directory.

---

## Usage

1. **Train the Model (Optional)**:
   - Use the `Train.ipynb` notebook to train the model on a dataset of your choice.
   - Save the trained model as `emotiondetector.h5`.

2. **Run the Web Application**:
   ```bash
   python test2.py
   ```
   Open your web browser and navigate to `http://localhost:5000` to access the application.

3. **Detect Emotions**:
   - Upload an image or use the webcam for real-time emotion detection.

---

## Project Structure

```
Real-Time-Emotion-Detection/
│
├── static/                     # Static assets (CSS, JS, Images)
├── templates/                  # HTML templates for Flask
├── Train.ipynb                 # Training script for the model
├── test2.py                    # Main Flask application script
├── emotiondetector.h5          # Pre-trained model
├── emotiondetector.json        # Model architecture
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## Future Scope
- Improve the model's accuracy by training on a larger dataset.
- Add support for additional emotions.
- Enhance the frontend design and user experience.
- Implement mobile-friendly responsiveness.
- Integrate with social media platforms for live emotion tracking.

---

## Contributors
- **Onkar Giri** - Repository owner and lead developer.
- **Asmita Teli** -Front-End Developer
- **Harshvardhan Killedar** - Documentation Head
- **Rohan Patil** - Overall Co-ordinator 


---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

