import math   # for sqrt
import random # for random sampling

# -----------------------------
# Part 1: k-means implementation
# -----------------------------
def get_distance(point1, point2):
    """
    Compute the Euclidean distance between two points.
    """
    # sum of squared differences
    square_sum = 0 # sum of squared differences
    #Iterate over each dimension of the two points together.
    for x, y in zip(point1, point2):
        square_sum += (x - y) ** 2  # add squared difference
    
    #return square root of square_sum
    return math.sqrt(square_sum)

def initialize_centroids(data, k, seed=2025):
    """
    Select k initial centroids randomly from the data.
    """

    #Set the random seed for reproducibility.
    random.seed(seed)

    #Number of data points.
    n = len(data)

    #Randomly choose k distinct indices from 0 to n-1.
    indices = random.sample(range(n), k)

    centroids = [] # List to store chosen centroids.

    #Use the selected indices to pick points from the data.
    for idx in indices:
        centroids.append(data[idx])

    return centroids

def assign_points(data, centroids):
    """
    Assign each data point to the nearest centroid.
    """
    assignments = [] # List to store cluster index for each point.

    # Iterate over all data points.
    for point in data:
        min_dist = float("inf") # Current minimum distance (start with infinity).

        closest_cluster = None # Index of the closest centroid.

        # Compare the point with each centroid.
        for cluster_idx, centroid in enumerate(centroids):
            dist = get_distance(point, centroid) # Use the distance function.

            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster_idx

        #After checking all centroids, assign the point to the closest cluster.
        assignments.append(closest_cluster)

    return assignments

def compute_new_centroids(data, assignments, k):
    """
    Compute new centroids as the mean of points in each other.
    """

    # Number of features (dimensions).
    num_dimensions = len(data[0])

    # Prepare lists to store sums and counts for each cluster.
    sums = [[0.0] * num_dimensions for _ in range(k)]
    counts = [0] * k

    # Accumulate sums and counts for each cluster.
    for point, cluster_idx in zip(data, assignments):
        counts[cluster_idx] += 1
        for d in range(num_dimensions):
            sums[cluster_idx][d] += point[d]

    # Compute the mean for each cluster.
    new_centroids = []

    for cluster_idx in range(k):
        if counts[cluster_idx] == 0:
            # If a cluster has no points, handle it in a simple way.
            new_centroids.append(random.choice(data))
        else:
            centroid = []
            for d in range(num_dimensions):
                mean_value = sums[cluster_idx][d] / counts[cluster_idx]

                centroid.append(mean_value)
            new_centroids.append(centroid)

    return new_centroids


def has_converged(old_centroids, new_centroids, tol=1e-6):
    """
    Check if centroids have stopped changing.
    """

    # Compare each old centroid with the corresponding new centroid.
    for c_old, c_new in zip(old_centroids, new_centroids):
        # Compare coordinated-by-coordinate within tolerance.
        for x, y in zip(c_old, c_new):
            if abs(x - y) > tol:
                # If difference is greater than tolerance -> not converged.
                return False
    
    # All coordinates are within tolerance -> converged.
    return True

def k_means(data, k, max_iters=100, seed=2025):
    """
    Run the k-means clustering algorithm.
    data : list of points (each point is a list of numbers)
    k : number of clusters
    max_iters : maximum number of iterations
    seed : random seed for centroid initialization
    """

    # 1. Initialize centroids.
    centroids = initialize_centroids(data, k , seed=seed)

    for iteration in range(max_iters):
        # 2. Assign each point to the nearest centroid.
        assignments = assign_points(data, centroids)

        # 3. Compute new centroids based on current assignments.
        new_centroids = compute_new_centroids(data, assignments, k)

        # 4. Check for convergence.
        if has_converged(centroids, new_centroids, tol=1e-6):
            centroids = new_centroids
            break

        # Update centroids for the next iterations.
        centroids = new_centroids

    return centroids, assignments

# -----------------------------
# Part 2: Apply algorithm to Tokyo Airbnb
# -----------------------------
def load_airbnb_data(csv_path):
    """
    Load Airbnb dataset from CSV and perform basing cleaning.
    """
    import pandas as pd

    # Read CSV file using pandas.
    df = pd.read_csv(csv_path)

    # Select only required columns.
    required_cols = ["latitude", "longitude", "price", "room_type"]
    df = df[required_cols].copy()
    
    # 1) Convert price values to string (in case of mixed numeric/string formats)
    df["price"] = df["price"].astype(str)

    # 2) Remove currency symbols ($, €, ₩ etc.) and commas
    df["price"] = df["price"].str.replace(r"[^0-9\.]", "", regex=True)

    # 3) Convert cleaned string to numeric (values that cannot be converted become NaN)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Drop rows that contain missing values in essential columns.
    df = df.dropna(subset=["latitude", "longitude", "price"])
    

    print("Number of rows after cleaning:", len(df))
    print("Columns:", df.columns)

    # Return cleaned Dataframe.
    return df

def run_clustering_on_airbnb(df, k=4):
    """
    Apply k-means clustering to Airbnb dataset.
    """
    print("Number of rows received in run_clustering_on_airbnb:", len(df))

    if len(df) < k:
        print(f"Warning: number of rows ({len(df)}) is smaller than k={k}.")
        print("Please check the data cleaning step or choose a smaller k.")
        return df  # 그냥 그대로 반환하고 k-means는 실행하지 않음

    # Convert Dataframe to plain Python list for k-means input.
    data_for_clustering = []

    for _, row in df.iterrows():
        # Use latitude, longitude, price as features.
        point = [row["latitude"], row["longitude"], row["price"]]
        data_for_clustering.append(point)
    
    print("Number of points in data_for_clustering:", len(data_for_clustering))

    # Run k-means algorithm (from Part 1).
    centroids, assignments = k_means(data_for_clustering, k=k, max_iters=100, seed=2025)

    # Add cluster assignment to Dataframe.
    df["cluster"] = assignments

    # Save result as a new CSV.
    output_path = "listings_with_clusters.csv"
    df.to_csv(output_path, index=False)

    # Return Dataframe with cluster column.
    return df

def main():
    import os
    os.makedirs("figures", exist_ok=True)

    # 1. Load CSV
    csv_path = "listings.csv"
    airbnb_df = load_airbnb_data(csv_path)

    # 2. k-means
    clustered_df = run_clustering_on_airbnb(airbnb_df, k=4)

    # Print check
    print(clustered_df.head())
    print("Clustering and CSV save complete!")


# -----------------------------
# Part 3: Analysis & plots
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt

def load_clustered_data(csv_path="listings_with_clusters.csv"):
    """
    Load the Airbnb Tokyo dataset that already includes cluster assignments.
    This file was generated in Part 2 after running k-means.
    """
    df = pd.read_csv(csv_path)
    return df


# 1) Why is one cluster more expensive than the others?
def analyze_prices_by_cluster(df):
    """
    Compute and print mean and median prices by cluster.
    Higher average prices may indicate:
    - desirable or central geographic location,
    - different composition of room types,
    - higher demand or tourist-heavy areas.
    """
    # Calculate mean price for each cluster
    mean_price = df.groupby("cluster")["price"].mean()

    # Calculate median price for each cluster
    median_price = df.groupby("cluster")["price"].median()

    print("Mean price by cluster:")
    print(mean_price)

    # Identify which cluster has the highest average price
    high_cluster = mean_price.idxmax()
    print(f"\nCluster with highest mean price: {high_cluster}")

    return mean_price, median_price, high_cluster


# 2) Do clusters differ in their composition of room types?
def analyze_room_type_by_cluster(df):
    """
    Count room types for each cluster.
    """
    # Count room types per cluster
    counts = pd.crosstab(df["cluster"], df["room_type"])

    print("\nRoom type counts by cluster:")
    print(counts)

    # Convert to proportions for better interpretation
    proportions = counts.div(counts.sum(axis=1), axis=0)

    print("\nRoom type proportions by cluster:")
    print(proportions)

    return counts, proportions


# 3) Scatter plot of all listings colored by cluster.
def plot_clusters(df):
    """
    This visualizes how clusters are geographically distributed in Tokyo.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(df["longitude"], df["latitude"], c=df["cluster"], s=5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Tokyo Airbnb Listings by Cluster (k-means)")
    plt.tight_layout()
    plt.savefig("figures/clusters_map.png", dpi=300, bbox_inches="tight")
    plt.show()


# 4) Highlight the cluster with the highest mean price.
def plot_high_price_cluster(df, high_cluster):
    """
    Other clusters are shown in light gray for comparison,
    making it easier to see where the high-price cluster is concentrated.
    """
    # Extract only listings from the highest-price cluster
    subset = df[df["cluster"] == high_cluster]

    plt.figure(figsize=(6, 6))

    # Plot all listings in light color
    plt.scatter(df["longitude"], df["latitude"], c="lightgray", s=5, label="Other clusters")

    # Plot the high-price cluster in red
    plt.scatter(
        subset["longitude"],
        subset["latitude"],
        c="red",
        s=8,
        label=f"Cluster {high_cluster} (highest mean price)"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("High-Price Cluster Location in Tokyo")
    plt.legend()
    plt.tight_layout()

    plt.savefig("figures/high_price_cluster.png", dpi=300, bbox_inches="tight") 
    plt.show()

if __name__ == "__main__":
    main()

    df_clustered = load_clustered_data("listings_with_clusters.csv")
    mean_price, median_price, high_cluster = analyze_prices_by_cluster(df_clustered)
    counts, proportions = analyze_room_type_by_cluster(df_clustered)

    plot_clusters(df_clustered)
    plot_high_price_cluster(df_clustered, high_cluster)