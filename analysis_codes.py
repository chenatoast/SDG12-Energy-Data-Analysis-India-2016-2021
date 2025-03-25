import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
from scipy.stats import norm
import scipy.stats as st

def load_data(filename):
    """Load CSV data into a pandas DataFrame"""
    df = pd.read_csv(filename)
    return df

def calculate_statistics(data, column_names):
    """Calculate mean and standard deviation for specified columns"""
    stats = {}
    for col in column_names:
        stats[f'mean_{col}'] = data[col].mean()
        stats[f'std_{col}'] = statistics.stdev(data[col])
    return stats

def z_test(data, col1, col2, n1=None, n2=None):
    """Perform Z-test between two columns"""
    if n1 is None:
        n1 = len(data[col1])
    if n2 is None:
        n2 = len(data[col2])
    
    mean1 = data[col1].mean()
    mean2 = data[col2].mean()
    std1 = statistics.stdev(data[col1])
    std2 = statistics.stdev(data[col2])
    
    z_value = (mean1 - mean2) / math.sqrt((std1**2 / n1) + (std2**2 / n2))
    
    return {
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'z_value': z_value
    }

def t_test(list1, list2):
    """Perform t-test between two lists"""
    mean1 = statistics.mean(list1)
    mean2 = statistics.mean(list2)
    std1 = statistics.pstdev(list1)
    std2 = statistics.pstdev(list2)
    
    # Calculate Sp (pooled standard deviation)
    Sp = math.sqrt((std1**2 + std2**2) / 2)
    
    # Calculate t-value
    t_value = (mean1 - mean2) / (Sp * math.sqrt(0.04 + 0.04))  # 0.04 is 1/25 as per original code
    
    return {
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'Sp': Sp,
        't_value': t_value
    }

def compare_with_critical_value(test_value, critical_value):
    """Compare test statistic with critical value and return hypothesis result"""
    if test_value > critical_value:
        return "Reject Null Hypothesis"
    else:
        return "Null Hypothesis Accepted"

def plot_bar_chart(data, x_column, y_column, title, color=None, figsize=(10, 5)):
    """Create a bar chart for the given data"""
    plt.figure(figsize=figsize)
    
    if isinstance(y_column, list):
        # Multiple columns to plot
        df1 = pd.DataFrame({col: list(data[col]) for col in y_column}, index=list(data[x_column]))
        ax = df1.plot.bar()
    else:
        # Single column
        df1 = pd.DataFrame({y_column: list(data[y_column])}, index=list(data[x_column]))
        ax = df1.plot.bar(color=color)
    
    plt.xlabel(x_column, size=15)
    plt.ylabel("Value", size=15)
    plt.title(title)
    plt.xticks(rotation=90)
    
    return plt

def plot_line_chart(data, x_column, y_columns, title, colors=None, figsize=(10, 5)):
    """Create a line chart for the given data"""
    plt.figure(figsize=figsize)
    
    if isinstance(y_columns, list):
        # Plot multiple lines
        for i, col in enumerate(y_columns):
            color = colors[i] if colors and i < len(colors) else None
            plt.plot(data[x_column], data[col], color=color, label=col)
        plt.legend()
    else:
        # Plot a single line
        plt.plot(data[x_column], data[y_columns], color=colors[0] if colors else None)
    
    plt.xlabel(x_column)
    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(rotation=90)
    
    return plt

def plot_normal_distribution(highlight_z=1.645):
    """Plot normal distribution with highlighted area"""
    x = np.arange(-5, 5, 0.001)
    plt.figure(figsize=(8, 5))
    plt.plot(x, norm.pdf(x, 0, 1), color='green')
    
    plt.ylabel('Density')
    plt.xlabel('z value')
    plt.title('Normal Distribution', fontsize=14)
    plt.fill_between(x, norm.pdf(x, 0, 1), where=(highlight_z < x) & (x < 4))
    
    return plt

def plot_scatter_correlation(data, x_column, y_column, title):
    """Create a scatter plot with correlation line"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column])
    
    # Add the regression line
    plt.plot(np.unique(data[x_column]), 
             np.poly1d(np.polyfit(data[x_column], data[y_column], 1))(np.unique(data[x_column])), 
             color='black')
    
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)
    
    # Calculate correlation coefficient
    corr = np.corrcoef(data[x_column], data[y_column])[0, 1]
    
    return plt, corr

def power_method(matrix, max_iterations=100, tolerance=1e-4):
    """Compute dominant eigenvalue and eigenvector using the power method"""
    n = matrix.shape[0]
    # Start with a vector of all ones
    x = np.ones((n, 1))
    
    eigen_values = [0]
    eigen_vectors = [x]
    
    i = 0
    while i < max_iterations:
        # Multiply matrix by the current vector
        y = np.matmul(matrix, eigen_vectors[i])
        
        # Find the largest absolute value in y
        eigen_value = np.max(np.abs(y))
        
        # Normalize the resulting vector
        y = np.round(y / eigen_value, 4)
        
        eigen_values.append(eigen_value)
        eigen_vectors.append(y)
        
        # Check for convergence
        if i > 0 and np.round(eigen_values[i+1], 4) == np.round(eigen_values[i], 4):
            break
            
        i += 1
    
    return {
        'eigenvalue': eigen_values[-1],
        'eigenvector': eigen_vectors[-1],
        'iterations': i,
        'all_values': eigen_values,
        'all_vectors': eigen_vectors
    }

def create_correlation_table(data, x_column, y_column, x_label="X", y_label="Y"):
    """Create a correlation analysis table"""
    df = pd.DataFrame()
    df[f'{x_label}'] = data[x_column]
    df[f'{y_label}'] = data[y_column]
    df[f'{x_label}^2'] = data[x_column]**2
    df[f'{y_label}^2'] = data[y_column]**2
    df[f'{x_label}{y_label}'] = data[x_column] * data[y_column]
    
    # Add the sum row
    sums = pd.DataFrame(df.sum()).T
    sums.index = ['Sigma']
    
    # Combine the data with the sum row
    df_with_sum = pd.concat([df, sums])
    
    return df_with_sum

def regression_analysis(data, y_column, time_range=None):
    """Perform regression analysis and predict future values"""
    if time_range is None:
        time_range = range(1, len(data) + 1)
    
    # Create dataframe for regression
    df = pd.DataFrame()
    df['Y'] = data[y_column]
    df['X'] = time_range
    df['XY'] = df['Y'] * df['X']
    df['X^2'] = df['X']**2
    
    # Calculate sums and means
    sums = pd.DataFrame(df.sum()).T
    sums.index = ['Sigma']
    
    means = pd.DataFrame(df.mean()).T
    means.index = ['Mean']
    
    # Combine the data with summary rows
    reg_data = pd.concat([df, sums, means])
    
    # Calculate beta (slope)
    numerator = reg_data['XY'].loc['Sigma'] - (reg_data['X'].loc['Sigma'] * reg_data['Y'].loc['Sigma']) / len(df)
    denominator = reg_data['X^2'].loc['Sigma'] - reg_data['X'].loc['Sigma']**2 / len(df)
    beta = numerator / denominator
    
    # Define the regression function
    def regression_function(x):
        return reg_data['Y'].loc['Mean'] + beta * (x - reg_data['X'].loc['Mean'])
    
    # Predict future values
    future_years = range(len(df) + 1, len(df) + 6)
    predictions = [regression_function(x) for x in future_years]
    
    return {
        'regression_data': reg_data,
        'beta': beta,
        'function': regression_function,
        'predictions': predictions,
        'prediction_years': future_years
    }

def plot_regression(regression_results, original_data, y_column, original_index=None, future_index=None):
    """Plot regression line with original and predicted values"""
    if original_index is None:
        original_index = original_data.index
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame({'Y': original_data[y_column]}, index=original_index)
    
    # Add predictions to the dataframe
    if future_index is not None and len(future_index) == len(regression_results['predictions']):
        for i, idx in enumerate(future_index):
            plot_df.loc[idx] = regression_results['predictions'][i]
    
    # Plot original and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df.index, plot_df['Y'], marker='o', linestyle='--', mfc='g', ms=7)
    
    # Plot the regression line
    xl = list(range(-1, len(original_data) + len(regression_results['predictions']) + 1))
    yl = [regression_results['function'](x+1) for x in xl]
    plt.plot(xl, yl)
    
    # Highlight future predictions
    pred_start = len(original_data)
    plt.scatter(xl[pred_start:pred_start+len(regression_results['predictions'])], 
                yl[pred_start:pred_start+len(regression_results['predictions'])], 
                color='red', s=200)
    
    plt.xlabel('\nYears', fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.title('Regression Line for Future Predictions', fontsize=14)
    plt.xticks(rotation=90)
    
    return plt

def calculate_consumption(population, per_capita_consumption, scale_factor=1000000):
    """Calculate consumption based on population and per capita consumption"""
    return (per_capita_consumption * population) / scale_factor

def compare_production_consumption(data, state_column, production_column, consumption_column):
    """Compare production and consumption across states/regions"""
    plt.figure(figsize=(12, 6))
    plt.plot(data[state_column], data[production_column], color='red', label='Production')
    plt.plot(data[state_column], data[consumption_column], color='green', label='Consumption')
    plt.legend()
    plt.xticks(rotation=90)
    plt.xlabel('State/Region')
    plt.ylabel('Units')
    plt.title('Comparison of Production and Consumption')
    
    return plt

def anova_analysis(data, columns):
    """Perform one-way ANOVA analysis"""
    # Calculate T (grand total)
    data['sigma_all'] = data[columns].sum(axis=1)
    T = data['sigma_all'].sum()
    
    # Calculate CF (correction factor)
    n = len(columns) * len(data)
    CF = (T**2) / n
    
    # Calculate SSB (sum of squares between groups)
    data['sig_all_sq'] = data['sigma_all']**2
    data['sig_x_sq_n'] = data['sig_all_sq'] / len(columns)
    SSB = data['sig_x_sq_n'].sum() - CF
    
    # Calculate SST (total sum of squares)
    column_squares = {}
    column_sums = {}
    for col in columns:
        col_name = f"{col}_sq"
        column_squares[col_name] = data[col]**2
        data[col_name] = data[col]**2
        column_sums[col] = data[col_name].sum()
    
    SST = sum(column_sums.values()) - CF
    
    # Calculate SSW (sum of squares within groups)
    SSW = SST - SSB
    
    # Degrees of freedom
    df_between = len(columns) - 1
    df_within = n - len(columns)
    df_total = n - 1
    
    # Mean squares
    MS_between = SSB / df_between
    MS_within = SSW / df_within
    
    # F-statistic
    F = MS_between / MS_within
    
    # p-value
    p_value = 1 - st.f.cdf(F, df_between, df_within)
    
    return {
        'SSB': SSB,
        'SSW': SSW,
        'SST': SST,
        'df_between': df_between,
        'df_within': df_within,
        'df_total': df_total,
        'MS_between': MS_between,
        'MS_within': MS_within,
        'F': F,
        'p_value': p_value
    }

# Example usage
if __name__ == "__main__":
    # Example for energy demand data
    demand_data = load_data('demand.csv')
    demand_stats = calculate_statistics(demand_data, ['2020-21', '2016-17'])
    z_test_result = z_test(demand_data, '2020-21', '2016-17')
    print(f"Z-test result: {z_test_result['z_value']}")
    
    # Example for correlation analysis
    demand_data = load_data('demand.csv')
    production_data = load_data('production.csv')
    
    # Create a plot to compare demand and production
    corr_plot, corr_value = plot_scatter_correlation(
        pd.DataFrame({'demand': demand_data['2020-21'], 'production': production_data['2020-21']}),
        'demand', 'production', 'Correlation between Demand and Production'
    )
    print(f"Correlation coefficient: {corr_value}")
    
    # Example for regression analysis
    reg_results = regression_analysis(demand_data, '2020-21')
    print(f"Regression predictions: {reg_results['predictions']}")