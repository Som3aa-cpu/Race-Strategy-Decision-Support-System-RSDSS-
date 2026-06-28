# Race Strategy Decision Support System (RSDSS)

## Project Overview

**Project Name:** Race Strategy Decision Support System (RSDSS)

### Goal

Develop an AI-powered software application that assists Formula 1 race strategy decisions by predicting tire degradation and determining the optimal pit stop strategy based on race conditions.

The project combines **Machine Learning**, **Data Analysis**, **Optimization**, and **Software Engineering** to simulate the work of a Formula 1 race strategy engineer.

Unlike a simple machine learning project that only predicts tire wear, the system will function as a complete engineering decision-support tool.

# Motivation

This project is intended to strengthen my portfolio for a Master's application in Automotive Engineering (particularly ESTACA).

It demonstrates skills across multiple engineering disciplines:

- Machine Learning
- Data Analysis
- Optimization Algorithms
- Software Development
- Vehicle Performance Analysis
- Engineering Decision Making

It also complements my other projects:

- CAD engine components designed in FreeCAD (piston, connecting rod, crankshaft)
- Future Embedded Systems projects
- Automotive software development

# High-Level System Architecture

Historical Formula 1 Data  
│  
▼  
Data Cleaning & Processing  
│  
▼  
Feature Engineering  
│  
▼  
Tire Degradation Prediction Model  
│  
Predicted tire wear for future laps  
│  
▼  
Race Strategy Optimization Engine  
│  
┌──────────────┬──────────────┬──────────────┐  
│ │ │ │  
Gap Analysis Pit Stop Model Tire Model Position Analysis  
│ │ │ │  
└──────────────┴──────────────┴──────────────┘  
│  
▼  
Recommended Race Strategy  
│  
▼  
Interactive User Dashboard

# Project Objectives

## Phase 1 - Data Collection

Collect historical Formula 1 race data.

Possible data sources:

- FastF1 Python library
- Ergast API (historical race information)
- Public Formula 1 datasets

Data may include:

- Lap times
- Sector times
- Tire compound
- Tire age
- Stint length
- Driver
- Team
- Circuit
- Position
- Gap to leader
- Gap to car ahead
- Gap to car behind
- Weather (if available)

## Phase 2 - Data Processing

Prepare the data by:

- Removing missing values
- Encoding categorical variables
- Normalizing numerical features
- Creating useful derived features

Example engineered features:

- Tire age
- Average pace over last 5 laps
- Tire compound
- Circuit characteristics
- Current stint length
- Tire performance trend

## Phase 3 - Tire Degradation Prediction

Train a machine learning model capable of predicting:

- Tire degradation
- Expected lap time
- Performance drop-off over future laps

Possible models:

- Random Forest
- XGBoost
- Gradient Boosting
- Neural Network (future version)

Outputs:

- Predicted lap time
- Estimated tire wear
- Remaining competitive tire life

# Race Strategy Optimization Engine

The optimization engine uses the ML predictions to determine the best pit strategy.

For every possible pit lap:

- Simulate pitting.
- Apply estimated pit lane time loss.
- Switch to a new tire compound.
- Predict future lap times.
- Compare with surrounding competitors.
- Estimate finishing position.

The optimizer selects the strategy that maximizes race performance.

# Inputs

Race:

- Circuit
- Total race laps
- Current lap

Driver:

- Driver
- Team
- Starting position

Current Race Situation:

- Current position
- Gap to car ahead
- Gap to car behind

Tires:

- Current compound
- Tire age

Optional (Future):

- Weather
- Safety Car probability
- Virtual Safety Car
- Rain probability

# Outputs

The application should provide:

- Recommended pit lap
- Recommended tire compound
- Predicted finishing position
- Estimated pit loss
- Predicted lap times after pit stop
- Tire degradation graph
- Strategy comparison

Example:

Recommended Strategy  
<br/>Pit Lap: 21  
<br/>Next Compound:  
Medium  
<br/>Estimated Pit Loss:  
21.8 s  
<br/>Expected Finish:  
P3  
<br/>Confidence:  
82%

# Optimization Logic

The optimizer should answer questions such as:

- Should I pit now?
- Should I extend the current stint?
- Will I rejoin ahead or behind another car?
- Is an undercut likely to succeed?
- Is an overcut more beneficial?
- Which tire compound offers the best race outcome?

# Assumptions (Version 1)

To keep the first version manageable:

- One pit stop strategy only
- Dry races only
- Standard pit stop duration per team
- Fixed pit lane loss for each circuit
- No Safety Car or Virtual Safety Car
- No mechanical failures
- No driver mistakes
- No changing weather

Future versions can gradually remove these simplifications.

# Technology Stack

Programming Language

- Python

Data Processing

- Pandas
- NumPy

Machine Learning

- Scikit-learn
- XGBoost (optional)

Visualization

- Plotly

Application

- Streamlit

Version Control

- Git
- GitHub

# Future Improvements

Potential future features include:

- Multi-stop strategy optimization
- Dynamic weather simulation
- Safety Car simulation
- Virtual Safety Car simulation
- Rain strategy optimization
- Fuel load estimation
- Monte Carlo race simulation
- Reinforcement Learning strategy engine
- Real-time race strategy updates
- Team-specific tire degradation models

# Learning Goals

Through this project I aim to improve my understanding of:

- Machine Learning
- Predictive Modeling
- Optimization Algorithms
- Data Engineering
- Python Software Development
- Engineering Decision Support Systems
- Motorsport Analytics
- Automotive Performance Analysis

# Why This Project Matters

This project demonstrates the ability to integrate several engineering disciplines into one cohesive system rather than producing a standalone machine learning model.

It showcases:

- Artificial Intelligence
- Software Engineering
- Systems Engineering
- Data Analysis
- Optimization
- Automotive Engineering concepts

The final result should resemble an engineering tool that could assist a Formula 1 race strategist rather than simply predicting numerical values.

# Long-Term Vision

The Race Strategy Decision Support System is intended to become the centerpiece of my engineering portfolio.

Combined with:

- Mechanical CAD projects (FreeCAD)
- Embedded Systems projects
- Future ECU software development

it will demonstrate a progression toward becoming an automotive software engineer with expertise spanning mechanical design, embedded systems, artificial intelligence, and engineering decision support.