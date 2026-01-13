# 2D Self-Driving Car Simulation (Neural Networks)

## Overview
This project implements a **2D self-driving car simulation** where an autonomous agent learns to navigate a road environment using **neural networks**.  
The agent perceives its surroundings through simulated sensors and makes driving decisions such as steering and acceleration.

The project demonstrates fundamental concepts of **autonomous systems**, including perception, decision-making, and control.

---

## Problem Statement
The objective is to train an agent that can:
- Stay within road boundaries
- Avoid collisions
- Drive smoothly for extended periods

This is formulated as a **learning-based control problem**.

---

## Approach
- Designed a 2D driving environment with road boundaries and obstacles
- Simulated distance-based sensor inputs using ray casting
- Used sensor readings as inputs to a neural network controller
- Trained the agent using reinforcement learning with a reward-driven approach
- Optimized behavior for safety and smooth navigation

---

## Neural Network
- Input layer: simulated sensor distance measurements
- Hidden layers: feature learning
- Output layer: steering and acceleration commands

The network learns a mapping from environment state to driving actions.

---

## Reinforcement Learning Setup
- **State:** Sensor readings and vehicle state
- **Actions:** Steering and acceleration
- **Reward Function:**
  - Positive reward for staying on the road
  - Penalty for collisions
  - Reward for smooth forward motion

---

## Evaluation
Performance was evaluated using:
- Collision frequency
- Distance traveled without crashing
- Stability of lane-following behavior

After training, the agent achieved **stable navigation with reduced collision rate**.

---

## Tech Stack
- Python
- NumPy
- Pygame
- Neural Networks
- Reinforcement Learning

---

## Project Structure
