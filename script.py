import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import requests
import folium
from streamlit_folium import folium_static

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_dim=24, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])

# Load the trained GRU model
MODEL_PATH = "C:/Raptee/ev_consumption_gru_model_full.pt"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# Extract model parameters
input_dim = checkpoint["input_dim"]
hidden_size = checkpoint["hidden_size"]
num_layers = checkpoint["num_layers"]
dropout = checkpoint["dropout"]

model = GRUModel(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Get coordinates using Open-Meteo API
def get_coordinates(place_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={place_name}&count=1&language=en&format=json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                location = data["results"][0]
                return location["latitude"], location["longitude"]
        return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coordinates: {e}")
        return None, None

def get_elevation(lat, lon):
    try:
        response = requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}")
        data = response.json()
        elevation = data.get("elevation", 0)
        return elevation[0] if isinstance(elevation, list) else elevation
    except:
        return 0

def get_routes(start_coords, end_coords):
    url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?alternatives=3&overview=full&geometries=geojson"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

# Streamlit UI
st.title("EV Energy Consumption Prediction")

# Input fields
start_place = st.text_input("Start Location (City, Address, or Landmark)")
end_place = st.text_input("End Location (City, Address, or Landmark)")

# Additional user inputs
time_of_day = st.slider("Time of Day (0-24 hours)", 0.0, 24.0, 12.0)
day_of_the_week = st.selectbox("Day of the Week", list(range(7)))
speed = st.number_input("Speed (km/h)", format="%.2f")
current = st.number_input("Current (A)", format="%.2f")
total_voltage = st.number_input("Total Voltage (V)", format="%.2f")
max_cell_temp = st.number_input("Max Battery Cell Temp (°C)", format="%.2f")
min_cell_temp = st.number_input("Min Battery Cell Temp (°C)", format="%.2f")
power_kw = st.number_input("Power (kW)", format="%.2f")
odometer = st.number_input("Odometer (km)", format="%.2f")
quantity_kwh = st.number_input("Quantity (kWh)", format="%.2f")
city = st.checkbox("City Road")
motor_way = st.checkbox("Motorway")
country_roads = st.checkbox("Country Roads")
driving_style = st.selectbox("Driving Style", [0, 1, 2], format_func=lambda x: ["Normal", "Moderate", "Fast"][x])
ecr_deviation = st.number_input("ECR Deviation", format="%.2f")
temperature = st.number_input("Environmental Temperature (°C)", format="%.2f")
percentage = st.number_input("Battery Percentage (%)", format="%.2f")
charging_time = st.number_input("Charging Time (minutes)", format="%.2f")
charge_energy = st.number_input("Charge Energy (kWh)", format="%.2f")

if st.button("Fetch Coordinates and Elevations"):
    start_lat, start_lon = get_coordinates(start_place)
    end_lat, end_lon = get_coordinates(end_place)
    
    if start_lat is not None and end_lat is not None:
        origin_elevation = get_elevation(start_lat, start_lon)
        destination_elevation = get_elevation(end_lat, end_lon)
        elevation_difference = destination_elevation - origin_elevation
        
        st.write(f"Origin Elevation: {origin_elevation:.2f} m")
        st.write(f"Destination Elevation: {destination_elevation:.2f} m")
        st.write(f"Elevation Difference: {elevation_difference:.2f} m")
        
        routes = get_routes((start_lat, start_lon), (end_lat, end_lon))
        
        if routes:
            best_routes = routes['routes'][:4]  # Ensure up to 4 routes are displayed
            st.write("### Best 4 Routes and Predicted Consumption")
            
            m = folium.Map(location=[start_lat, start_lon], zoom_start=12)
            colors = ["blue", "green", "red", "purple"]  # Different colors for different paths
            
            for i, route in enumerate(best_routes):
                distance = route['distance'] / 1000  # Convert to km
                trip_time_length = distance / speed * 60
                
                road_type = [int(city), int(motor_way), int(country_roads)]
                input_features = torch.tensor([[
                    time_of_day, day_of_the_week, origin_elevation, destination_elevation, elevation_difference, speed, current, total_voltage,
                    max_cell_temp, min_cell_temp, trip_time_length, power_kw, odometer, quantity_kwh, *road_type,
                    driving_style, ecr_deviation, distance, temperature, percentage, charging_time, charge_energy
                ]], dtype=torch.float32).unsqueeze(0)  # Fix input shape
                
                with torch.no_grad():
                    predicted_consumption = model(input_features).item()
                
                st.write(f"Route {i+1}: {distance:.2f} km, Predicted Consumption: {predicted_consumption:.2f} kWh")
                
                route_coords = [(lat, lon) for lon, lat in route['geometry']['coordinates']]
                folium.PolyLine(route_coords, color=colors[i % len(colors)], weight=5, opacity=0.7).add_to(m)
            
            folium_static(m)
    else:
        st.error("Could not fetch coordinates. Please check the place names.")