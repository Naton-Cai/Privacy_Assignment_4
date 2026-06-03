import numpy as np
import matplotlib.pyplot as plt
from diffprivlib.mechanisms import Exponential

BUCKETS = list(range(1, 101))

def plot_histogram(x_values, y_values=None, bins=20, title="Histogram Plot", xlabel="X Values", ylabel="Y Values"):
    """
    Plots a histogram using provided x and y values.

    Parameters:
    - x_values: List of x-axis values (bins or categories).
    - y_values: List of y-axis values (frequency or density values).
    - bins: Number of bins for the histogram (default: 50, controls bar width).
    - title: Title of the plot (default: "Histogram Plot").
    - xlabel: Label for the x-axis (default: "X Values").
    - ylabel: Label for the y-axis (default: "Y Values").

    Returns:
    - A displayed histogram plot.
    """
    # If x_values are actual data points, use plt.hist()
    if len(x_values) > bins:
        plt.hist(x_values, bins=bins, edgecolor='black')
    else:
        # If x_values are bin edges/categories, use plt.bar()
        plt.bar(x_values, y_values, width=(max(x_values) - min(x_values)) / bins, edgecolor='black')

def age_utility(true_age: int, bucket: str) -> float:
    return (100 - abs(true_age - bucket))/ 100

def sanitize_age(true_age: int, epsilon: float = 1.0) -> str:
    scores = [age_utility(true_age, b) for b in BUCKETS]
    mech = Exponential(
        epsilon=epsilon,
        sensitivity=0.1,
        utility=scores,
        candidates=BUCKETS,
    )
    return mech.randomise()


sampleSize = 1000
for age in [5, 24, 40, 67]: 
    current_age_noise_array = [] # Initialize for each age
    print(f"\n  Age {age}")
    age_noise_array = []
    
    for eps in np.arange(0.1, 3.0, 0.1):
      resultSum = 0
      for i in range(sampleSize): # simulate across 1000 trials
        resultSum += abs(sanitize_age(age, epsilon=eps)-age)
      result = resultSum / sampleSize
      age_noise_array.append(result)
    plot_histogram(np.arange(0.1, 3.0, 0.1), age_noise_array, bins= 30, title=f"Mean Noise for Age {age} (Sample Size: {sampleSize})", xlabel="Epsilon", ylabel="Noise")