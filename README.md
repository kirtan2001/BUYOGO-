# BUYOGO-
# 🏨 Hotel Booking Analytics & Q&A API

This project is a **FastAPI-based Q&A system** that provides hotel booking analytics and answers booking-related queries using FAISS indexing and NLP.

---

## Features
Predicts revenue, cancellations, and ADR from hotel bookings 
Embeds booking data using FAISS for fast retrieval  
REST API built with FastAPI  
Endpoints for analytics and Q&A  

---

## Installation & Setup

1️⃣ Clone the Repository
```sh
git clone https://github.com/kirtan2001/BUYOGO-.git
cd BUYOGO-


pip install -r requirements.txt

## Fast API server
## uvicorn app:app --reload
## swaggerUI : http://127.0.0.1:8000/docs


Return booking analytics 
{
  "cancellation_rate": "25.6%",
  "average_revenue": "1385.45",
  "average_adr": "104.67"
}
2) /ask (POST)
{
  "question": "What is the total revenue for July 2017?"
}


{
  "answer": {
    "hotel": "City Hotel",
    "arrival_date": "2017-07-17",
    "revenue": 1520.01,
    "cancellation": 0
  }
}


