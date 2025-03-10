import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Generate synthetic time series data
start_date = datetime(2025, 2, 21)
days = 30
dates = [start_date + timedelta(days=i) for i in range(days)]
traditional_progress = np.cumsum(np.random.uniform(0.3, 1.0, days))
recursive_progress = np.cumsum(np.random.uniform(2.0, 4.5, days))

# Initialize figure and axis
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(15, 8))

# Enhanced Plot Styling
ax.plot(dates, traditional_progress, label="Traditional Math Course", color='#FF5733', linewidth=3, linestyle='dashed', marker='o', markersize=8, alpha=0.85)
ax.plot(dates, recursive_progress, label="The Course That Calculates Itself", color='#1F77B4', linewidth=4, linestyle='solid', marker='D', markersize=8, alpha=0.9)

# Highlighting the Difference
ax.fill_between(dates, traditional_progress, recursive_progress, where=(recursive_progress > traditional_progress), interpolate=True, color='#1F77B4', alpha=0.3, label="Recursive Learning Advantage")
ax.fill_between(dates, traditional_progress, recursive_progress, where=(traditional_progress > recursive_progress), interpolate=True, color='#FF5733', alpha=0.3, label="Traditional Learning Advantage")

# Improve readability and aesthetics
ax.set_xlabel("Date", fontsize=18, fontweight='bold', color='darkblue')
ax.set_ylabel("Cumulative Learning Progress", fontsize=18, fontweight='bold', color='darkblue')
ax.set_title("Advanced Time Series Comparison: Traditional vs Recursive Learning", fontsize=20, fontweight='bold', pad=25, color='darkred')
ax.legend(loc="upper left", frameon=True, fontsize=14, fancybox=True, shadow=True)

# Tweak axes
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Grid and final touch
ax.grid(color='gray', linestyle='--', linewidth=0.6, alpha=0.8)
plt.show()