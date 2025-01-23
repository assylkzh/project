# Smart traffic light management system project

This README provides an overview of the  Smart traffic light management system project. 

Problem Statement: 
Nowadays, urban areas are  facing increasing traffic jams problems due to growing populations and proportionally, vehicle numbers which lead to longer travel times. And since current traffic management systems are often operated on pre-set signal timings, the city transport systems are now significantly inefficient. 


## Project Description

The Smart Traffic Management System is a system that uses predictions based on Machine Learning and traffic light adjustment algorithms to optimise city traffic flow in real-time. The logic of a system is that it will dynamically adjust traffic light timings depending on a current traffic situation on the road that will be predicted using ML models. 

## Project overview
Overall, the project implemented 2 models  for Traffic volume prediction: 1- using real-life (approaches_prediction.ipynb), real-time data that was collected manually using API in Astana city, and 2 - using synthetic open-source data that was collected from the Kaggle database(traffic_volume_prediction). Furthermore, 2 traffic light adjustment algorithms were created (one fully working and integrated with ML part,  and the second one is the demo version,working without ML).

## About Data

The data for the project were collected from 4 sources. All the referenes are given bellow. 
- **TomTom API Traffic Flow Data** 
- **TomTom API Junction Data** 
- **Historical Weather API Data from Open-Meteo**
- **Kaggle Dataset : "traffic-volume-dataset"**

## File Structure

- **`approaches_prediction.ipynb`**: Prediction for the traffic volume using Real-Life data collected in Astana using TomTom Api and merged with weather data from Open-Meteo API. 
- **`traffic_volume_prediction`**:  Prediction for the traffic volume using synthetic open-source collected from the Kaggle database(traffic_volume_prediction)
- **Python Script**: The main script performs data preprocessing and visualization.

## Prerequisites

- Python 3.8+
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `sklearn`
- Django==3.2.5

Install the required libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data and Database 
Fistly, I preprocessed the data from '2024_10_10_11_36_33_definition.csv' csv file.

The explanation of the features of the csv file:

id -   id is related to each separate approach and exit of our junction.
type - connected to the "id" column, it indicates which segmented pathway it's related to, the approach
of our junction, or one its exits.
name - the name of each approach and exit, it also contains information about the road direction.
roadName - the name of the road where each approach and exit lines are located.
direction - information about the direction of the traffic flow on a given approach or exit.
frc - information about the road class. One of the values: 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7. The lower this number
is, the bigger the road it's related to (0 being a motorway).
length - information about the length of a given approach (in meters).
oneWayRoad - indicates if the approach or exit is a one-way road (if the value is "False" then it means
the road is bi-directional)
excluded – indicates if the junction approach was excluded from the results or not (for example disabled
via toggle in the web application).
driveable - this value provides information if this specific road section (the approach or the exit) is
classified as "drivable" on our map.
dataNotAvailable – in case there is something wrong with any of the approaches (errors in processing
the data), this value will be set to “True” and there won’t be any data for it.

From this file I retrieved all columns, and created 'traffic-system' database in PostgreSQL. Then, created the table called 'traffic-main' and imported all the columns and rows to the 'traffic-main' table.



The next is 'approaches.csv' file.

Data structure explanation:

approachID – a unique ID number assigned to this specific approach line
travelTimeSec - information on how long (in seconds) it currently takes for the vehicles to travel through the approach
freeFlowTravelTime - information on how long (in seconds) it takes for the vehicles to travel through the approach during free-flow conditions( when there is no traffic jams, etc)
delaySec - the current delay (in seconds)
usualDelaySec - this is the usual delay that is expected at this time of day
Stops - the average number of stops per vehicle on the approach
queueLengthMeters - information on how long (in meters) the current line of vehicles waiting on the junction approach is
volumePerHour - an estimated value of how many vehicles we observe traveling through the approach per hour
isClosed – information if the approach was closed for traffic
stopsHistogram - a container that stores histogram data specific to stops ( example: 0,3 is displaying the data like this: "numberOfStops": 0, "numberOfVehicles": 3)

For this data I created 'approaches' table in the same database (traffic-system) and imported all the data from the csv to the table.

##


## Referenes for data 
1.	https://www.tomtom.com/products/tomtom-move/ 
2.	https://open-meteo.com/ 
3.	https://www.kaggle.com/datasets/rohith203/traffic-volume-dataset 
