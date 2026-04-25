
# 🤖 AI-Powered Customer Support Ticket Classifier

This project is an AI-driven system that automatically classifies customer support messages into predefined categories and assigns priority levels.

---

## 🚀 Features

- Classifies messages into:
  - Billing
  - Technical Issue
  - Account
  - General Inquiry

- Assigns priority levels:
  - High
  - Medium
  - Low

- Uses OpenAI API for intelligent classification  
- Includes a fallback keyword-based classifier when API is unavailable  
- Returns structured JSON output  
- Implements error handling and logging  

---

## 🧠 Approach

1. Accept a list of customer messages  
2. Send each message to OpenAI API using a structured prompt  
3. Parse and validate JSON response  
4. If API fails, switch to fallback keyword-based classification  
5. Return results in structured JSON format  

---

## ⚙️ Tech Stack

- Python
- OpenAI API
- dotenv (for environment variables)
- logging module

---

## 📌 Sample Input

```python
[
  "My payment got deducted but service is not activated",
  "App crashes every time I login",
  "How to change my email address?"
]
