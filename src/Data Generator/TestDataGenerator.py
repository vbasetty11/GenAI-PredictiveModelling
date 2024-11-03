import random
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Define the number of synthetic transactions to generate
num_transactions = 50000

# Define the list of possible values for each column
countries = ['Germany', 'France', 'Austria', 'Spain']
psps = ['PSP1', 'PSP2', 'PSP3', 'PSP4', 'PSP5']
card_providers = ['Master', 'Visa', 'Gold', 'Platinum']

# Generate synthetic data
data = []
for _ in range(num_transactions):
    transaction = {
        'tmsp': fake.date_time_this_year(),
        'country': random.choice(countries),
        'amount': round(random.uniform(5.0, 500.0), 2),
        'success': random.choice([0, 1]),
        'PSP': random.choice(psps),
        '3D_secured': random.choice([0, 1]),
        'card': random.choice(card_providers)
    }
    data.append(transaction)

# Introduce retries for a few failed transactions
retry_data = []
for transaction in data:
    retry_data.append(transaction)
    if transaction['success'] == 0 and random.random() < 0.3:  # 30% chance to retry a failed transaction
        num_retries = random.randint(1, 3)
        for _ in range(num_retries):
            retry_transaction = transaction.copy()
            retry_transaction['tmsp'] = fake.date_time_between_dates(datetime_start=transaction['tmsp'], datetime_end=transaction['tmsp'] + pd.Timedelta(minutes=1))
            retry_data.append(retry_transaction)

# Create a DataFrame from the synthetic data with retries
df = pd.DataFrame(retry_data)

# Sort the DataFrame by timestamp in ascending order
df = df.sort_values(by='tmsp')

# Save the DataFrame to a CSV file
df.to_csv('synthetic_transactions.csv', index=False)

print("Synthetic data generated with retries for a few failed transactions, sorted by timestamp, and saved to "
      "synthetic_transactions.csv")

