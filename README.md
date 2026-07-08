# Neural Network Image Recognition from Scratch 2023 (R)
# Overview

This project implements a feed-forward neural network in R to classify handwritten digit images. The project was developed independently following completion of a statistics degree and coursework in data mining, with the goal of strengthening understanding of the mathematical foundations and computational implementation of neural networks.

Rather than relying on existing machine learning frameworks, the model was implemented from first principles using matrix operations and fundamental concepts including forward propagation, activation functions, cost calculation, backpropagation, and gradient descent optimization.

# Motivation

The project was inspired by an honours-level introductory data mining course completed during my statistics degree, where neural networks and machine learning methods were introduced as analytical techniques for pattern recognition and predictive modeling.

Following the course, I independently developed this implementation to further explore how neural networks function internally, focusing on the mathematical principles behind model training, parameter optimization, and prediction.

# Dataset

The model was trained and tested using a publicly available handwritten digit image dataset consisting of standardized 28x28 pixel images.

Each observation represents a grayscale image where pixel intensity values are used as input features, with the corresponding digit label used as the target output.

# Model Architecture

The implemented neural network consists of:

Input layer:
Normalized image pixel features
Hidden layer:
Fully connected layer
ReLU activation function
Output layer:
Digit classification output
Implementation Details

The neural network was implemented manually using mathematical principles underlying machine learning models, including:

Weight and bias initialization
Forward propagation
Activation functions
Cost function calculation
Backpropagation using gradient calculations
Gradient descent optimization
Model prediction and assessment
Data Processing

The preprocessing pipeline includes:

Randomized dataset shuffling
Training/testing data split
Feature normalization
Removal of near-zero variance features
Label encoding for classification

# Purpose

The purpose of this project was to develop practical understanding of machine learning algorithms by implementing the underlying mathematics directly.

The project demonstrates knowledge of how neural networks learn through transformations, activation functions, error calculation, and iterative parameter optimization rather than only applying pre-built machine learning libraries.

# Technologies

R
Matrix-based numerical computation
Statistical computing
Machine learning fundamentals
