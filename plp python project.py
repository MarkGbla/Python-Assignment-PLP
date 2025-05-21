# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
sns.set(style="whitegrid")
plt.style.use('seaborn')

# Task 1: Load and Explore the Dataset
# --------------------------------------------------
try:
    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✅ Dataset loaded successfully")
    print("\nFirst 5 rows of the dataset:")
    display(iris_df.head())
    
    # Dataset information
    print("\n📊 Dataset information:")
    print(iris_df.info())
    
    # Check for missing values
    print("\n🔍 Missing values per column:")
    print(iris_df.isnull().sum())
    
    # Data cleaning (though Iris dataset is clean)
    if iris_df.isnull().sum().sum() > 0:
        print("\n🧹 Cleaning missing values...")
        # Fill numerical columns with mean
        num_cols = iris_df.select_dtypes(include=['float64']).columns
        iris_df[num_cols] = iris_df[num_cols].fillna(iris_df[num_cols].mean())
        print("Missing values after cleaning:", iris_df.isnull().sum().sum())
    else:
        print("\n✨ Dataset is already clean - no missing values found")

except Exception as e:
    print(f"❌ Error loading dataset: {str(e)}")
    raise

# Task 2: Basic Data Analysis
# --------------------------------------------------
print("\n📈 Basic statistics of numerical columns:")
display(iris_df.describe().transpose())

# Group by species and compute mean
print("\n🌷 Mean values by species:")
species_stats = iris_df.groupby('species').mean()
display(species_stats)

# Interesting findings
print("\n🔎 Key observations:")
print("1. Setosa has the smallest petal dimensions (length: 1.46cm, width: 0.25cm)")
print("2. Virginica has the largest measurements overall (petal length: 5.55cm)")
print("3. Versicolor is intermediate in most measurements")
print("4. Sepal width shows least variation across species (2.9-3.4cm range)")

# Task 3: Data Visualization
# --------------------------------------------------
plt.figure(figsize=(16, 12))

# Visualization 1: Line chart (measurement trends by species)
plt.subplot(2, 2, 1)
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for species in iris_df['species'].unique():
    species_data = iris_df[iris_df['species'] == species][features].mean()
    plt.plot(features, species_data, marker='o', label=species, linewidth=2)
plt.title('📊 Measurement Trends by Iris Species', pad=20, fontsize=14)
plt.ylabel('Measurement (cm)', fontsize=12)
plt.xlabel('Feature', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Species')
plt.grid(True, alpha=0.3)

# Visualization 2: Bar chart (average petal length by species)
plt.subplot(2, 2, 2)
sns.barplot(x='species', y='petal length (cm)', data=iris_df, ci=None, 
            palette='viridis', estimator='mean')
plt.title('📏 Average Petal Length by Species', pad=20, fontsize=14)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.xlabel('Species', fontsize=12)
plt.ylim(0, 6)

# Visualization 3: Histogram (sepal width distribution)
plt.subplot(2, 2, 3)
sns.histplot(data=iris_df, x='sepal width (cm)', kde=True, 
             bins=15, color='skyblue', edgecolor='black')
plt.title('📊 Distribution of Sepal Width', pad=20, fontsize=14)
plt.xlabel('Sepal Width (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Visualization 4: Scatter plot (sepal length vs petal length)
plt.subplot(2, 2, 4)
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', 
                hue='species', palette='Set2', s=100, alpha=0.8)
plt.title('🔍 Sepal Length vs Petal Length', pad=20, fontsize=14)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.legend(title='Species')

plt.tight_layout(pad=3.0)
plt.suptitle('Iris Dataset Analysis', y=1.02, fontsize=16)
plt.show()