# SmartSpend Mumbai

AI-powered Mumbai price intelligence dashboard helping users decide whether to BUY NOW or WAIT based on live market conditions.

## Live Demo
https://smartspend-1k5s.onrender.com

## Features
- Price prediction system
- BUY NOW / WAIT recommendations
- Groq NLP Text-to-SQL querying
- Statistical modeling
- Interactive dashboard
- Real-time insights

## Tech Stack
- Python
- Flask
- Streamlit
- Scikit-learn
- Groq API
- Pandas
- Plotly

## Screenshots
<img width="1919" height="897" alt="image" src="https://github.com/user-attachments/assets/98f1d151-083d-4f1a-9ad4-2856dc6f3114" />
<img width="1841" height="364" alt="image" src="https://github.com/user-attachments/assets/b8415320-7a89-40cd-8a6f-4f162b08c91b" />
<img width="1913" height="859" alt="image" src="https://github.com/user-attachments/assets/eef202e7-bcb9-4486-a16a-ac0822508e8b" />
<img width="1900" height="703" alt="image" src="https://github.com/user-attachments/assets/e53fd2b4-9869-454c-8d67-1679a58074d5" />
<img width="1857" height="552" alt="image" src="https://github.com/user-attachments/assets/c11ea132-1e16-45c3-a72a-fbd1eb3cce40" />


## How It Works

SmartSpend Mumbai is an AI-powered price intelligence platform designed to help users decide whether to BUY NOW or WAIT based on market behavior across essential household categories in Mumbai.

The system analyzes 16,000+ real-world price data points across categories including vegetables, medicines, petrol, flights, hotels and groceries.

Workflow:

1. **Data Collection & Processing**
   - Historical pricing datasets are cleaned, structured and standardized using Pandas and NumPy.
   - Time-series trends and category-wise patterns are extracted for modeling.

2. **Statistical & ML Analysis**
   - Multiple statistical techniques including STL Decomposition, Markov Chains, MANOVA and Logistic Regression are applied to understand price movement behavior.
   - Separate prediction models are trained per category to generate BUY NOW or WAIT recommendations.

3. **Natural Language AI Querying**
   - Users can ask plain-English questions such as:
     > “Which vegetable is cheapest this month?”
   - Groq LLM API converts these queries into structured database operations using Text-to-SQL logic.

4. **Interactive Dashboard**
   - The frontend visualizes predictions, trends and category insights through an interactive dashboard.
   - Real-time recommendation outputs help users make smarter purchase decisions.


