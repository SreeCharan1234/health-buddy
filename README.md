# health-buddy
![image](https://github.com/user-attachments/assets/bb222e97-0e93-4893-83e5-9cd793c500c6)

![image](https://github.com/user-attachments/assets/dad86ad8-4d75-408f-9a38-7f3be26d3078)

![image](https://github.com/user-attachments/assets/bd383aff-0578-4da1-ae7e-253a6fc70131)


![image](https://github.com/user-attachments/assets/0faafd7a-4120-45d0-b538-1910e6ad20f8)
![image](https://github.com/user-attachments/assets/9dc8f3ab-a4d5-4550-a5d1-29fdad460539)
![image](https://github.com/user-attachments/assets/541d6612-40e6-451c-bfa4-f0a36d3ec3fa)
![image](https://github.com/user-attachments/assets/2f38f423-9609-4c62-b32f-822a59701b94)
![image](https://github.com/user-attachments/assets/5d196b1f-0303-4129-be48-69ed52dba328)










# Heart Disease Prediction and Notification System

## Project Overview

This project is a medical alert system that predicts the risk of heart disease and notifies the patient via email. Using advanced machine learning techniques and agent-based message passing, this system evaluates a patient's health data for potential heart disease risks and automates the process of delivering this result to the patient.

## Table of Contents
1. [Project Importance](#project-importance)
2. [Problems Addressed by uAgents](#problems-addressed-by-uagents)
3. [Technologies Used](#technologies-used)
4. [How This Project Is Helpful](#how-this-project-is-helpful)
5. [Code Walkthrough](#code-walkthrough)
6. [Future Enhancements](#future-enhancements)

---

## Project Importance

Heart disease remains a leading cause of death worldwide, and early diagnosis is critical for effective treatment. This project leverages machine learning to assess a user's health indicators, potentially identifying early signs of heart disease. By using agents, the process of prediction and notification is automated, reducing delays in patient communication and allowing patients to receive prompt guidance on their health status.

## Problems Addressed by uAgents

- **Agent-Based Interaction**: By implementing agents, this project creates a distributed architecture where each agent has a specific role—one handles prediction and the other manages notifications.
- **Automated Notification**: Manual notification is time-consuming and prone to error. The notification agent handles email alerts automatically, ensuring patients are promptly informed of their health status.
- **Data Privacy**: uAgents allow for secure communication between agents, reducing the need for intermediary systems and minimizing data exposure.
- **Scalability**: By using agents, this system can scale with ease. Additional agents can be added to handle more patients or introduce new functionalities without overloading the existing structure.

## Technologies Used

1. **uAgents Framework**: Provides the base for defining and handling independent agents (`alice` and `bob`).
2. **Joblib**: Used for loading the pre-trained heart disease prediction model.
3. **Pandas**: Handles the input data, transforming it into a structured format for the prediction model.
4. **Mailjet API**: Sends emails with the prediction result to notify the patient.
5. **Machine Learning Model**: A trained model for predicting heart disease risk using patient health data.

## How This Project Is Helpful

1. **Early Detection**: This system helps in the early detection of heart disease, potentially saving lives by informing patients early.
2. **Efficiency in Healthcare**: Automating diagnosis and notification increases the efficiency of healthcare providers by reducing manual work.
3. **Convenience for Patients**: Patients receive notifications directly in their email, ensuring they are promptly informed without needing to check for results manually.
4. **Enhanced Data Security**: By using agents for message passing, patient data is shared in a secure and controlled manner, which is especially important in healthcare applications.

## Code Walkthrough

### 1. Agent Definitions
Two agents are defined in the system:
- **Alice (Prediction Agent)**: Handles the prediction of heart disease risk using a trained machine learning model.
- **Bob (Notification Agent)**: Manages notification delivery to the patient via email.

### 2. Data Collection
The system collects key patient information:
```python
user_input = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
```
Each input is transformed into a Pandas DataFrame format, suitable for the model.

### 3. Heart Disease Prediction
Using a pre-trained model (`heart_new.pkl`), the system predicts whether the patient is at risk of heart disease.
```python
prediction=model.predict(data)
```

### 4. Message Passing
If the prediction indicates a high risk, `alice` sends a message to `bob` via uAgent communication channels.

### 5. Patient Notification
`bob` sends an email to the patient with the prediction result using the Mailjet API:
```python
data = {
    'FromEmail': 'your_email@example.com',
    'FromName': 'Your Name',
    'Subject': 'Your test result',
    'Text-part': f'dear patient {msg.text}',
    'Recipients': [{'Email': 'patient_email@example.com'}]
}
```

## Future Enhancements

- **Multi-Agent Communication**: Expand to more agents for additional functionalities, such as scheduling appointments or prescribing medications.
- **Data Visualization**: Provide patients with visual feedback, such as graphs and charts.
- **Mobile App Integration**: Create a companion mobile app for real-time notifications.
- **Extended Health Monitoring**: Extend the system to monitor additional health indicators for broader diagnoses.

---
![1](https://github.com/user-attachments/assets/9ae84987-034e-4f3f-a65b-80513a5f071b)
![2](https://github.com/user-attachments/assets/ce38bde6-7d30-48fd-9cf3-35e10fe2217e)
![3](https://github.com/user-attachments/assets/0f9da708-ed2f-4552-899f-1ca55945fe90)
![4](https://github.com/user-attachments/assets/c84df30d-d442-4963-9f1c-8e1f481d79c5)
![5](https://github.com/user-attachments/assets/5c5967c6-a1bb-444d-a9f6-63b91b2a1486)
![6](https://github.com/user-attachments/assets/fd016951-dba9-40ad-87a3-be266991c929)
![7](https://github.com/user-attachments/assets/f7c9350d-233e-4c7c-a08e-4aa5d3ff55b7)

![8](https://github.com/user-attachments/assets/10746afe-9ccb-4907-8ec1-6b0d0c6aead0)
![9](https://github.com/user-attachments/assets/b646ebe3-282e-47bd-a8ee-7cd5a2924015)
![10](https://github.com/user-attachments/assets/2e5e0313-08d9-498e-bb21-c79c49752078)
![11](https://github.com/user-attachments/assets/78fa2334-31e4-479e-9d0d-a8b6250c4cb6)
