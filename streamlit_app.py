import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

st.set_page_config(page_title="Bank Loan AI", page_icon="🏦")
st.title("🏦 Bank Loan AI - Sujatha")
st.write("AI predicts if your loan will be APPROVED or REJECTED")

# --- Train AI with sample bank data ---
# Rule: High salary + High credit score + Small loan = APPROVED
data = {
    'salary': [30000, 50000, 80000, 25000, 100000, 40000, 60000, 20000, 90000, 35000],
    'loan_amount': [200000, 300000, 500000, 400000, 400000, 500000, 300000, 300000, 600000, 200000],
    'credit_score': [750, 700, 800, 550, 820, 620, 720, 500, 780, 650],
    'approved': [1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
}
df_train = pd.DataFrame(data)
X = df_train[['salary', 'loan_amount', 'credit_score']]
y = df_train['approved']
model = LogisticRegression()
model.fit(X, y)

# --- User Input ---
st.subheader("Enter Your Details")
name = st.text_input("Your Name")
salary = st.number_input("Monthly Salary (Rs)", min_value=10000, max_value=500000, value=50000)
loan_amount = st.number_input("Loan Amount Wanted (Rs)", min_value=50000, max_value=2000000, value=300000)
credit_score = st.slider("Credit Score (300-900)", 300, 900, 700)

if st.button("Check Loan Approval"):
    if name == "":
        st.warning("Please enter your name")
    else:
        pred = model.predict([[salary, loan_amount, credit_score]])[0]
        result = "APPROVED" if pred == 1 else "REJECTED"

        if pred == 1:
            st.success(f"🎉 {name} - APPROVED ✅")
            st.balloons()
            st.write(f"Salary: Rs {salary}, Loan: Rs {loan_amount}, Credit Score: {credit_score}")
        else:
            st.error(f"❌ {name} - REJECTED")
            st.write("Tip: Increase salary or credit score, or ask for smaller loan.")

        # Save to CSV
        new_row = pd.DataFrame([{
            'Name': name, 'Salary': salary, 
            'Loan_Amount': loan_amount, 'Credit_Score': credit_score,
            'Result': result
        }])
        if os.path.exists("results.csv"):
            new_row.to_csv("results.csv", mode='a', header=False, index=False)
        else:
            new_row.to_csv("results.csv", index=False)
        st.success("Saved!")

# --- Show All Applications ---
st.subheader("All Loan Applications")
if os.path.exists("results.csv"):
    df = pd.read_csv("results.csv")
    st.dataframe(df)
    with open("results.csv", "rb") as f:
        st.download_button("📥 Download results.csv", data=f, file_name="results.csv", mime="text/csv")
else:
    st.write("No applications yet. Be the first!")
