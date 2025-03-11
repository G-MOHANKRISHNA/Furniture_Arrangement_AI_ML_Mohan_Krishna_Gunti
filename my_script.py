import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Define the AI Model
class LayoutNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LayoutNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate Synthetic Training Data
def generate_data(samples=100):
    data = []
    labels = []
    for _ in range(samples):
        room_w, room_h = np.random.randint(5, 15, size=2)
        furniture_w, furniture_h = np.random.randint(1, min(room_w, room_h), size=2)
        x = np.random.uniform(0, room_w - furniture_w)
        y = np.random.uniform(0, room_h - furniture_h)
        data.append([room_w, room_h, furniture_w, furniture_h])
        labels.append([x, y])
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

# Train Model
def train_model():
    model = LayoutNet(4, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    X_train, Y_train = generate_data(500)
    X_train, Y_train = torch.tensor(X_train), torch.tensor(Y_train)
    
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), "layout_model.pth")
    return model

# Load trained model
def load_model():
    model = LayoutNet(4, 2)
    model.load_state_dict(torch.load("layout_model.pth"))
    model.eval()
    return model

# Visualize layout
def visualize_layout(room_w, room_h, furniture_w, furniture_h, x, y):
    fig, ax = plt.subplots()
    ax.set_xlim(0, room_w)
    ax.set_ylim(0, room_h)
    ax.add_patch(plt.Rectangle((x, y), furniture_w, furniture_h, color='blue', alpha=0.5))
    ax.set_title("Optimized Furniture Layout")
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("Furniture Layout Optimizer")
    room_w = st.number_input("Room Width", min_value=5, max_value=20, value=10)
    room_h = st.number_input("Room Height", min_value=5, max_value=20, value=10)
    furniture_w = st.number_input("Furniture Width", min_value=1, max_value=10, value=3)
    furniture_h = st.number_input("Furniture Height", min_value=1, max_value=10, value=3)
    model = load_model()
    if st.button("Generate Layout"):
        input_data = torch.tensor([[room_w, room_h, furniture_w, furniture_h]], dtype=torch.float32)
        x, y = model(input_data).detach().numpy()[0]
        visualize_layout(room_w, room_h, furniture_w, furniture_h, x, y)

if __name__ == "__main__":
    train_model()
    main()
