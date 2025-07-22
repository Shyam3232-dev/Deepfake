from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
import cv2
import os
from werkzeug.utils import secure_filename
from torchvision import models

app = Flask(__name__)

# Model Architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# Instantiate the model
model = Model(num_classes=2)  # Adjust num_classes based on your specific use case

# Load the state_dict into the model
model.load_state_dict(torch.load('model/model_84_acc_10_frames_final_data.pt', map_location=torch.device('cpu')))
model.eval()

# Function to extract frames from video
def extract_frames(video_path, sequence_length):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % sequence_length == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

# Function to preprocess frames (example preprocessing, adjust as necessary)
def preprocess(frames):
    # Example: Resize and normalize frames
    processed_frames = [cv2.resize(frame, (224, 224)) for frame in frames]
    processed_frames = torch.tensor(processed_frames).permute(0, 3, 1, 2).float() / 255.0
    return processed_frames.unsqueeze(0)  # Adjust dimensions if needed

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)
            
            sequence_length = int(request.form.get("sequence_length", 20))
            frames = extract_frames(file_path, sequence_length)
            if len(frames) > 0:
                processed_frames = preprocess(frames)
                with torch.no_grad():
                    _, output = model(processed_frames)
                    prediction = output.argmax(dim=1).item()  # Example output processing
                    result = "REAL" if prediction == 1 else "FAKE"  # Map prediction to labels
            else:
                result = "No frames extracted"
            
            os.remove(file_path)  # Clean up the uploaded file
            return render_template("result.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)

