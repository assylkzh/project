# Smart traffic light management system project

This README provides an overview of the  Smart traffic light management system project. 

Problem Statement: 
Nowadays, urban areas are  facing increasing traffic jams problems due to growing populations and proportionally, vehicle numbers which lead to longer travel times. And since current traffic management systems are often operated on pre-set signal timings, the city transport systems are now significantly inefficient. 


## Project Description

The Smart Traffic Management System is a system that uses predictions based on Machine Learning and traffic light adjustment algorithms to optimise city traffic flow in real-time. The logic of a system is that it will dynamically adjust traffic light timings depending on a current traffic situation on the road that will be predicted using ML models. 

## Project overview
Overall, the project implemented model for Traffic volume prediction using real-life - real-time data that was collected manually using API in Astana city. Furthermore, 2 traffic light adjustment algorithms were created (one fully working and integrated with ML part,  and the second one is the demo version,working without ML).

## About Data

The data for the project were collected from 4 sources. All the referenes are given bellow. 
- **TomTom API Traffic Flow Data** 
- **TomTom API Junction Data** 
- **Historical Weather API Data from Open-Meteo**

## File Structure

- **`approaches_prediction.ipynb`**: Prediction for the traffic volume using Real-Life data collected in Astana using TomTom Api and merged with weather data from Open-Meteo API. 

## Prerequisites

- Python 3.8+
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `sklearn`
  - `psycopg2` 
  - `sqlalchemy` 

Install the required libraries using:
```bash
pip install pandas numpy scikit-learn sqlalchemy psycopg2 matplotlib
```

## Data and Database 

From data collected using TomTom Junction and Traffic Flow APIs I retrieved all columns, and created 'traffic-system' database in PostgreSQL. Then, created the tables called 'traffic-main' and 'approaches',  imported all collected data. 
Data from both tables were cleaned and merged. 

## Model 
I split the data into training and testing sets (80% for training, 20% for testing) and trained RandomForestRegressor Model 

## Results 
The RandomForestRegressor achieved an R2 score of 0.7559 on the test data, indicating a good fit.

## Referenes for data 
1.	https://www.tomtom.com/products/tomtom-move/ 
2.	https://open-meteo.com/ 
3.	https://www.kaggle.com/datasets/rohith203/traffic-volume-dataset 
