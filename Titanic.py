import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('titanic.csv')

# Basic info
print("=== BASIC INFO ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Survival rate: {df['Survived'].mean()*100:.1f}%")

# Survival by gender
print("\n=== SURVIVAL BY GENDER ===")
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
print(gender_survival)

# Survival by class
print("\n=== SURVIVAL BY CLASS ===")
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
print(class_survival)

# Simple plots
plt.figure(figsize=(12, 4))

# Plot 1: Gender survival
plt.subplot(1, 3, 1)
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Gender')

# Plot 2: Class survival
plt.subplot(1, 3, 2)
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival by Class')

# Plot 3: Age distribution
plt.subplot(1, 3, 3)
df['Age'].hist(bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')

plt.tight_layout()
plt.show()

# Age groups
df['AgeGroup'] = pd.cut(df['Age'], [0, 18, 35, 60, 100], labels=['Child', 'Young', 'Adult', 'Senior'])
print("\n=== SURVIVAL BY AGE GROUP ===")
age_survival = df.groupby('AgeGroup')['Survived'].mean() * 100
print(age_survival)

# Fare vs survival
print("\n=== FARE VS SURVIVAL ===")
print(f"Average fare - Survived: ${df[df['Survived']==1]['Fare'].mean():.2f}")
print(f"Average fare - Died: ${df[df['Survived']==0]['Fare'].mean():.2f}")

# Key findings
print("\n=== KEY FINDINGS ===")
print("1. Women survived more than men")
print("2. 1st class had highest survival rate")
print("3. Children had better survival chances")
print("4. Higher fare = better survival chance")