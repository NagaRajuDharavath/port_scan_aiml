# README – Port Scan Classification Using Machine Learning (Simple Version)

## 1. Project Overview
This project detects different types of port scans using Machine Learning. Port scanning is used by attackers to find open ports on a target.  
The model classifies traffic into:
- SYN Scan
- FIN Scan
- Xmas Scan
- UDP Scan
- Normal Traffic

## 2. Purpose
The goal is to show how Machine Learning can help detect port scanning activity by learning patterns from network flow features.

## 3. Files in This Project
- portscan_dataset.csv → dataset used for training
- portscan_ml.py → Python script containing the ML model
- README.md → documentation file

## 4. Dataset Details
The dataset contains:
- flow_duration  
- total_packets  
- packet_size_mean  
- syn_flag  
- fin_flag  
- rst_flag  
- psh_flag  
- urg_flag  
- protocol (6 = TCP, 17 = UDP)  
- label (Normal, SYN, FIN, UDP, Xmas)

Each row is one flow of network traffic.

## 5. How to Run the Project

### Step 1: Install Python
Download Python from https://www.python.org/downloads/  
Check the box “Add Python to PATH”.

### Step 2: Install required libraries
Run:
```
pip install pandas scikit-learn seaborn matplotlib
```

### Step 3: Go to project folder
```
cd C:\PortScanProject```

### Step 4: Run the script
```
python portscan_ml.py
```

You will see:
- Accuracy
- Classification report
- Confusion matrix graph

## 6. How the Code Works (Simple)
1. Loads the CSV dataset  
2. Cleans the data  
3. Encodes labels  
4. Trains a Random Forest model  
5. Tests the model  
6. Shows accuracy and confusion matrix  

## 7. Conclusion
This mini project shows that Machine Learning can detect different types of port scans with high accuracy.

