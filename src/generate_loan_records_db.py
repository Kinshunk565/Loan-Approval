"""
Loan Records Database Generator
================================
Generates a synthetic database of 5,000 people with individual
loan-level records for the records lookup feature.

Produces two CSVs:
  - loan_records_people.csv  (person-level data)
  - loan_records_history.csv (individual loan records)

Author: Kinshunk Garg
"""

import pandas as pd
import numpy as np
import os

# Realistic Indian first and last names
FIRST_NAMES_MALE = [
    "Rahul", "Amit", "Vikram", "Suresh", "Rajesh", "Arun", "Deepak", "Sanjay",
    "Manoj", "Nitin", "Rohit", "Ajay", "Prashant", "Vivek", "Anand", "Sachin",
    "Gaurav", "Ashish", "Pankaj", "Manish", "Karan", "Arjun", "Varun", "Rohan",
    "Mohit", "Vishal", "Kunal", "Harsh", "Shubham", "Akash", "Tushar", "Ravi",
    "Naveen", "Sunil", "Aman", "Tarun", "Dinesh", "Ramesh", "Rakesh", "Yogesh",
    "Abhishek", "Ankur", "Hitesh", "Jatin", "Lalit", "Mukesh", "Neeraj",
    "Piyush", "Sumit", "Vinod"
]

FIRST_NAMES_FEMALE = [
    "Priya", "Neha", "Anjali", "Pooja", "Sneha", "Kavita", "Sunita", "Meena",
    "Ritu", "Swati", "Divya", "Shruti", "Anita", "Rekha", "Nisha", "Komal",
    "Pallavi", "Sonal", "Deepa", "Rashmi", "Shweta", "Aarti", "Geeta", "Sapna",
    "Mansi", "Tanvi", "Isha", "Megha", "Richa", "Kriti", "Bhavna", "Chitra",
    "Durga", "Ekta", "Gauri", "Heena", "Jyoti", "Kiran", "Lata", "Maya",
    "Namita", "Padma", "Radha", "Sita", "Tara", "Uma", "Vandana", "Yamini",
    "Zara", "Aditi"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Joshi", "Agarwal",
    "Mishra", "Reddy", "Nair", "Iyer", "Mehta", "Shah", "Chopra", "Malhotra",
    "Kapoor", "Bhatia", "Saxena", "Thakur", "Yadav", "Chauhan", "Pandey",
    "Dubey", "Shukla", "Tiwari", "Dwivedi", "Srivastava", "Rawat", "Bhatt",
    "Jain", "Bansal", "Garg", "Goyal", "Mittal", "Rastogi", "Khanna", "Bose",
    "Das", "Mukherjee", "Chandra", "Pillai", "Menon", "Rao", "Naidu",
    "Kulkarni", "Deshmukh", "Patil", "Pawar", "Deshpande"
]

LOAN_TYPES = [
    "Home Loan", "Personal Loan", "Car Loan", "Education Loan",
    "Business Loan", "Gold Loan", "Two-Wheeler Loan", "Consumer Durable Loan"
]

LENDERS = [
    "SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "PNB",
    "Bank of Baroda", "Kotak Mahindra", "IndusInd Bank",
    "Yes Bank", "Canara Bank", "Union Bank", "IDBI Bank",
    "Bajaj Finserv", "Tata Capital", "LIC Housing Finance"
]

STATUSES = ["Repaid", "Defaulted", "Active"]


def generate_loan_records_db(n_people: int = 5000, seed: int = 42):
    """Generate a synthetic loan records database."""
    np.random.seed(seed)

    people_rows = []
    records_rows = []
    record_id_counter = 1

    for i in range(1, n_people + 1):
        pid = f"PER{str(i).zfill(5)}"

        # Gender: 60% Male, 40% Female
        gender = np.random.choice(["Male", "Female"], p=[0.60, 0.40])
        if gender == "Male":
            first = np.random.choice(FIRST_NAMES_MALE)
        else:
            first = np.random.choice(FIRST_NAMES_FEMALE)
        last = np.random.choice(LAST_NAMES)
        full_name = f"{first} {last}"

        age = int(np.clip(np.random.normal(35, 10), 21, 65))
        phone = f"9{np.random.randint(100000000, 999999999)}"

        people_rows.append({
            "Person_ID": pid,
            "Full_Name": full_name,
            "Gender": gender,
            "Age": age,
            "Phone": phone,
        })

        # Number of past loans: 0-8, most have 1-4
        n_loans = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            p=[0.08, 0.18, 0.25, 0.20, 0.13, 0.08, 0.04, 0.03, 0.01]
        )

        # Creditworthiness randomly assigned (affects repayment)
        creditworthiness = np.random.choice(
            ["excellent", "good", "fair", "poor"],
            p=[0.25, 0.35, 0.25, 0.15]
        )

        for _ in range(n_loans):
            rid = f"REC{str(record_id_counter).zfill(6)}"
            record_id_counter += 1

            loan_type = np.random.choice(LOAN_TYPES, p=[
                0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03
            ])

            # Loan amount based on type
            if loan_type == "Home Loan":
                amount = int(np.clip(np.random.lognormal(5.2, 0.5), 50, 800))
            elif loan_type == "Personal Loan":
                amount = int(np.clip(np.random.lognormal(3.5, 0.6), 5, 100))
            elif loan_type == "Car Loan":
                amount = int(np.clip(np.random.lognormal(4.0, 0.4), 10, 200))
            elif loan_type == "Education Loan":
                amount = int(np.clip(np.random.lognormal(3.8, 0.5), 10, 150))
            elif loan_type == "Business Loan":
                amount = int(np.clip(np.random.lognormal(4.5, 0.7), 20, 500))
            elif loan_type == "Gold Loan":
                amount = int(np.clip(np.random.lognormal(2.5, 0.5), 2, 50))
            else:
                amount = int(np.clip(np.random.lognormal(2.0, 0.6), 1, 30))

            # Loan date: random in past 15 years
            year = np.random.randint(2010, 2026)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            loan_date = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"

            # Loan term
            if loan_type == "Home Loan":
                term = np.random.choice([120, 180, 240, 300, 360])
            elif loan_type in ["Car Loan", "Education Loan"]:
                term = np.random.choice([36, 48, 60, 84])
            else:
                term = np.random.choice([12, 24, 36, 48, 60])

            # Status based on creditworthiness
            if creditworthiness == "excellent":
                status = np.random.choice(STATUSES, p=[0.75, 0.03, 0.22])
            elif creditworthiness == "good":
                status = np.random.choice(STATUSES, p=[0.60, 0.10, 0.30])
            elif creditworthiness == "fair":
                status = np.random.choice(STATUSES, p=[0.45, 0.25, 0.30])
            else:  # poor
                status = np.random.choice(STATUSES, p=[0.25, 0.45, 0.30])

            lender = np.random.choice(LENDERS)

            records_rows.append({
                "Record_ID": rid,
                "Person_ID": pid,
                "Loan_Type": loan_type,
                "Loan_Amount": amount,
                "Loan_Date": loan_date,
                "Loan_Term_Months": term,
                "Status": status,
                "Lender": lender,
            })

    people_df = pd.DataFrame(people_rows)
    records_df = pd.DataFrame(records_rows)

    return people_df, records_df


def summarize_person_records(records_df: pd.DataFrame, person_id: str) -> dict:
    """
    Summarize a person's loan records into the fields needed for prediction.

    Returns:
        dict with: prev_loan_count, prev_loans_repaid, prev_loan_defaults,
                   avg_prev_loan_amount, repayment_rate, records_table
    """
    person_records = records_df[records_df["Person_ID"] == person_id].copy()

    if len(person_records) == 0:
        return {
            "prev_loan_count": 0,
            "prev_loans_repaid": 0,
            "prev_loan_defaults": 0,
            "avg_prev_loan_amount": 0,
            "repayment_rate": 0.0,
            "records_table": pd.DataFrame(),
        }

    # Only count completed loans (Repaid or Defaulted) for history stats
    completed = person_records[person_records["Status"].isin(["Repaid", "Defaulted"])]
    total_completed = len(completed)
    repaid = len(completed[completed["Status"] == "Repaid"])
    defaulted = len(completed[completed["Status"] == "Defaulted"])
    avg_amount = round(person_records["Loan_Amount"].mean(), 1)
    repayment_rate = round(repaid / total_completed, 2) if total_completed > 0 else 0.0

    return {
        "prev_loan_count": total_completed,
        "prev_loans_repaid": repaid,
        "prev_loan_defaults": defaulted,
        "avg_prev_loan_amount": avg_amount,
        "repayment_rate": repayment_rate,
        "records_table": person_records.sort_values("Loan_Date", ascending=False),
    }


if __name__ == "__main__":
    print("Generating loan records database...")
    people_df, records_df = generate_loan_records_db(5000)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    people_path = os.path.join(data_dir, "loan_records_people.csv")
    records_path = os.path.join(data_dir, "loan_records_history.csv")

    people_df.to_csv(people_path, index=False)
    records_df.to_csv(records_path, index=False)

    print(f"✅ People database saved:  {people_path} ({len(people_df)} people)")
    print(f"✅ Records database saved: {records_path} ({len(records_df)} records)")
    print(f"\n   Avg loans per person: {len(records_df) / len(people_df):.1f}")
    print(f"   People with 0 loans:  {(people_df['Person_ID'].isin(records_df['Person_ID'].unique()) == False).sum()}")

    # Show status distribution
    print(f"\n   Status Distribution:")
    for status, count in records_df["Status"].value_counts().items():
        print(f"     {status}: {count} ({count / len(records_df) * 100:.1f}%)")

    print(f"\n   Sample People:")
    print(people_df.head(10).to_string(index=False))
    print(f"\n   Sample Records:")
    print(records_df.head(10).to_string(index=False))
