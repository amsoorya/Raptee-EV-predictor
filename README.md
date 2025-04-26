# EV Energy Consumption Prediction

![EV Energy Consumption](https://github.com/jayasoorya/ev-consumption-prediction/raw/main/assets/banner.png)

## 🔋 Overview

EV Energy Consumption Prediction is a web application built with Streamlit that helps electric vehicle (EV) users predict energy consumption for different routes. The application uses a trained GRU (Gated Recurrent Unit) neural network model to provide accurate energy consumption estimates based on various parameters including route characteristics, driving conditions, and vehicle state.

This project was developed for **CodeVolt'25** conducted by Zoho and VIT.

## ✨ Features

- **Route Selection**: Enter start and end locations to get the best possible routes
- **Multiple Route Analysis**: Compare up to 4 alternative routes with their predicted energy consumption
- **Interactive Map**: Visualize routes on an interactive map powered by Folium
- **Comprehensive Parameter Inputs**: Consider factors like:
  - Time of day and day of the week
  - Driving style and road types
  - Vehicle speed and battery status
  - Environmental conditions
  - Elevation differences

## 🛠️ Technology Stack

- **Python** - Core programming language
- **PyTorch** - Deep learning framework for the GRU model
- **Streamlit** - Web application framework
- **Folium** - Interactive map visualization
- **Open-Meteo API** - Geocoding and elevation data
- **OSRM API** - Route planning and navigation

## 📋 Prerequisites

- Python 3.7+
- PyTorch
- Streamlit
- Internet connection for API access

## ⚙️ Installation

1. Clone this repository:
   ```
   git clone https://github.com/jayasoorya/ev-consumption-prediction.git
   cd ev-consumption-prediction
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the trained model file (`ev_consumption_gru_model_full.pt`) in the correct location as specified in the code.

## 🚀 Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter start and end locations, adjust parameters according to your vehicle and driving conditions

4. Click "Fetch Coordinates and Elevations" to see route options and predicted energy consumption

## 📊 Model Details

The prediction model uses a Gated Recurrent Unit (GRU) neural network architecture with the following specifications:
- Input features: 24 parameters including road conditions, vehicle status, and environmental factors
- Hidden size: Configurable, default 64
- Number of layers: Configurable, default 2
- Dropout rate: Configurable, default 0.2

## 📈 Future Improvements

- Real-time traffic integration
- User accounts to save preferred routes and vehicle profiles
- Mobile application development
- Integration with EV charging station databases
- Support for more vehicle models with specific consumption patterns

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/jayasoorya/ev-consumption-prediction/issues).

## 👤 Contributors

- **Jaya Soorya** - [GitHub Profile](https://github.com/amsoorya)
  - Contact: amjayasoorya@gmail.com

## 📝 License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.

## 🙏 Acknowledgements

- Zoho and VIT for organizing CodeVolt'25
- Open-Meteo for providing free geocoding and elevation APIs
- Project OSRM for route planning capabilities
