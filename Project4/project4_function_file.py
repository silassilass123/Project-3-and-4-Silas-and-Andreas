from __future__ import annotations
"""
Created on Fri Nov 21 09:08:37 2025

@author: silas
"""

"""
milkyway_clustering.py

Utility functions for Project 4

The idea is that the Jupyter notebook should only orchestrate the workflow
and call the functions defined here.

This file does not execute anything by itself when imported.
"""


from typing import Tuple, Literal
from types import SimpleNamespace
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from astropy import units as u
from mw_plot import MWSkyMap
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd



EncodingName = Literal["brightness", "rgb", "normalized_rgb"]

# Task 0 
def SkyMapConfig(
    center: str = "M31",
    radius_arcsec: Tuple[float, float] = (8800.0, 8800.0),
    background: str = "Mellinger color optical survey",
    figsize: Tuple[float, float] = (5.0, 5.0),
):
    """
    Create a simple configuration object for a Milky Way sky map.

    Usage (same as before, but now it's a function):
        cfg = SkyMapConfig(center="M31", radius_arcsec=(8800.0, 8800.0))

    The returned object has attributes:
        cfg.center, cfg.radius_arcsec, cfg.background, cfg.figsize
    """
    return SimpleNamespace(
        center=center,
        radius_arcsec=radius_arcsec,
        background=background,
        figsize=figsize,
    )


def SamplingConfig(step: int = 4):
    """
    Configuration for sampling pixels from an RGB image.

    Usage:
        cfg = SamplingConfig(step=4)

    The returned object has attribute:
        cfg.step
    """
    return SimpleNamespace(step=step)



# Plotting and conversion utilities


def create_mw_skymap(config=None):
    """
    Create a Milky Way sky map using MWSkyMap.

    Parameters
    ----------
    config:
        SkyMapConfig-like object with attributes:
        center, radius_arcsec, background, figsize.
        If None, a default configuration is used (center="M31").

    Returns
    -------
    fig, ax, mw :
        Matplotlib figure and axis with the map, and the MWSkyMap instance.
    """
    if config is None:
        config = SkyMapConfig()

    mw = MWSkyMap(
        center=config.center,
        radius=(config.radius_arcsec[0], config.radius_arcsec[1]) * u.arcsec,
        background=config.background,
    )

    fig, ax = plt.subplots(figsize=config.figsize)
    mw.transform(ax)
    ax.set_title(f"Milky Way sky map – center: {config.center}")

    return fig, ax, mw

    
def generate_sky_maps_for_coords(coords, radii):
    """
    Generate Milky Way sky maps for different centres and radii.

    Parameters
    ----------
    coords : list
        List of SkyCoord objects (e.g. [m31, sgr, polaris]).
    radii : list of tuple
        List of (radius_x, radius_y) pairs in arcseconds.

    Returns
    -------
    figs : list
        List of generated matplotlib figures.
    """
    figs = []

    for coord, r in zip(coords, radii):
        lon = coord.ra.deg
        lat = coord.dec.deg

        # same wrap logic as in your notebook
        
#Task 3
def figure_to_rgb_array(fig):
    """
    Convert a Matplotlib figure to an RGB numpy array of shape (H, W, 3).
    """
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()

    # Modern Matplotlib uses buffer_rgba()
    buf = np.asarray(fig.canvas.buffer_rgba())

    # Drop alpha channel → keep RGB
    rgb_array = buf.reshape((height, width, 4))[:, :, :3]

    return rgb_array




# Encoding of pixels
def sample_pixels(rgb_array: np.ndarray, cfg=None):
    """
    Sub-sample pixels from an RGB image on a regular grid.

    Parameters

    rgb_array:
        Image array of shape (H, W, 3).
    cfg:
        SamplingConfig-like object with attribute 'step'; if None, uses step=4.

    Returns
    
    coords:
        Array of shape (N, 2) with (row, col) indices for each sampled pixel.
    colors:
        Array of shape (N, 3) with the corresponding RGB values.
    """
    if cfg is None:
        cfg = SamplingConfig()

    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb_array must have shape (H, W, 3).")

    h, w, _ = rgb_array.shape
    step = max(1, int(cfg.step))

    rows = np.arange(0, h, step)
    cols = np.arange(0, w, step)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    coords = np.column_stack((rr.ravel(), cc.ravel()))
    colors = rgb_array[rr, cc].reshape(-1, 3)

    assert coords.shape[0] == colors.shape[0]
    return coords, colors


def encode_brightness(colors: np.ndarray) -> np.ndarray:
    """
    Encode pixels using a single brightness value (grey level).

    Parameters
    
    colors:
        Array of shape (N, 3) with RGB values in [0, 255].

    Returns
    
    features:
        Array of shape (N, 1) with brightness values.
    """
    if colors.shape[1] != 3:
        raise ValueError("colors must have shape (N, 3).")

    weights = np.array([0.299, 0.587, 0.114])
    grey = colors @ weights
    return grey.reshape(-1, 1)


def encode_rgb(colors: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Use the raw (or normalized) RGB channels as features.

    Parameters:
    colors:
        Array of shape (N, 3) with RGB values.
    normalize:
        If True, remove brightness by normalizing each pixel by (R+G+B),
        making the features represent relative color ratios.

    Returns
    features:
        Array of shape (N, 3).
    """
    if colors.shape[1] != 3:
        raise ValueError("colors must have shape (N, 3).")

    # Convert to float
    colors_f = colors.astype(float)

    if not normalize:
        # Raw RGB (brightness included)
        return colors_f

    # Improved normalized RGB (chromaticity) 
    # Scale to [0,1]
    colors_f /= 255.0

    # Sum R+G+B for each pixel
    s = colors_f.sum(axis=1, keepdims=True)

    # Avoid division by zero
    s[s == 0] = 1.0

    # Return color ratios (brightness removed)
    return colors_f / s



def build_features(
    rgb_array: np.ndarray,
    encoding: str = "brightness",
    sampling_cfg=None,
):
    """
    High-level helper: sample an image and create features for clustering.

    Parameters
    
    rgb_array:
        Image array of shape (H, W, 3).
    encoding:
        One of "brightness", "rgb", "normalized_rgb".
    sampling_cfg:
        Sampling configuration (SamplingConfig-like).

    Returns
    
    coords:
        Pixel coordinates of shape (N, 2).
    features:
        Encoded features of shape (N, D).
    """
    coords, colors = sample_pixels(rgb_array, sampling_cfg)

    if encoding == "brightness":
        features = encode_brightness(colors)
    elif encoding == "rgb":
        features = encode_rgb(colors, normalize=False)
    elif encoding == "normalized_rgb":
        features = encode_rgb(colors, normalize=True)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return coords, features


# Clustering
def kmeans_cluster(
    features: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 0,
):
    """
    Run K-means clustering on the provided features.

    Parameters

    features:
        Array of shape (N, D) with feature vectors.
    n_clusters:
        Number of clusters to use.
    random_state:
        Random seed for reproducibility.

    Returns
    
    labels:
        Array of shape (N,) with integer cluster labels.
    model:
        The fitted sklearn KMeans instance.
    """
    
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(features)

    assert labels.shape[0] == features.shape[0]
    return labels, km


def overlay_clusters_on_image(
    rgb_array: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    figsize: Tuple[float, float] = (6.0, 6.0),
    marker_size: float = 5.0,
):
    """
    Plot the original image and overlay a scatter plot of clustered pixels.

    Parameters
    
    rgb_array:
        Image array of shape (H, W, 3).
    coords:
        Pixel coordinates (row, col) of shape (N, 2).
    labels:
        Cluster labels for each sampled pixel, shape (N,).
    figsize:
        Size of the resulting figure.
    marker_size:
        Size of the scatter markers.

    Returns
    
    fig, ax, scatter:
        Matplotlib figure, axis and scatter plot handle.
    """
    if coords.shape[0] != labels.shape[0]:
        raise ValueError("coords and labels must have the same length.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_array)
    scatter = ax.scatter(
        coords[:, 1],  # x = column index
        coords[:, 0],  # y = row index
        c=labels,
        s=marker_size,
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_axis_off()
    ax.set_title("Clustered pixels overlaid on original image")

    return fig, ax, scatter

# Topic 2

# Task 0
# Ebola epidemic plots from project 2, exercise 5
def step_rk4(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def step_euler(f, y, t, dt):
    return y + dt * f(y, t)

def solve_ode(f, y0, tspan, dt, method='rk4'):
    y0 = np.asarray(y0, float)
    t0, t1 = tspan
    n = int(np.ceil((t1 - t0) / dt))
    t = np.linspace(t0, t1, n + 1)
    y = np.zeros((n + 1, y0.size))
    y[0] = y0

    step = step_rk4 if method == 'rk4' else step_euler
    for i in range(n):
        y[i + 1] = step(f, y[i], t[i], t[i + 1] - t[i])

    return t, y

def load_country(path: str):
    # Read ONLY the 3rd column (index 2) as float, skip the 1-line header.
    # Whitespace-delimited files like:
    # Date  Cases  New_Cases
    # 2013-12-30  86  4
    new_cases = np.genfromtxt(
        path,
        usecols=(2,),      # third column only
        skip_header=1,     # skip "Date Cases New_Cases"
        dtype=float,
        encoding="utf-8",
    )

    new_cases = np.atleast_1d(new_cases).astype(float)
    t = np.arange(new_cases.size, dtype=float)  # day index
    cum = np.cumsum(new_cases)
    return t, new_cases, cum
gamma = 1.0 / 7.0      # removal rate (infectious period ~7 days)
sigma = 1.0 / 9.7      # incubation (~9.7 days)
N = 1.2e7              # demo population (adjust if needed)

def f_SEZR_ebola(y, t, beta0, lam, N, sigma=sigma, gamma=gamma):
    S, E, Z, R = y
    bt = beta0 * math.exp(-lam * t)
    dS = -bt * S * Z / N
    dE =  bt * S * Z / N - sigma * E
    dZ =  sigma * E - gamma * Z
    dR =  gamma * Z
    return np.array([dS, dE, dZ, dR], float)

def simulate_ebola(beta0, lam, t_days, Z0=1.0):
    y0 = np.array([N - Z0, 0.0, Z0, 0.0], float)
    t_num, Y = solve_ode(
        lambda y, tt: f_SEZR_ebola(y, tt, beta0, lam, N),
        y0,
        (float(t_days[0]), float(t_days[-1])),
        dt=1.0,
        method="rk4",
    )
    S, E, Z, R = Y.T
    return t_num, R

def plot_country_data(country_data):
    """
    Plot new and cumulative Ebola cases for each country.
    """
    for country, (t, new_cases, cum) in country_data.items():
        
        # Plot new cases
        plt.figure()
        plt.plot(t, new_cases, label="New cases")
        plt.xlabel("Time [days]")
        plt.ylabel("New cases")
        plt.title(f"Exercise 5 — {country}: New cases")
        plt.legend()
        plt.show()

        # Plot cumulative cases
        plt.figure()
        plt.plot(t, cum, label="Cumulative cases")
        plt.xlabel("Time [days]")
        plt.ylabel("Cumulative cases")
        plt.title(f"Exercise 5 — {country}: Cumulative cases")
        plt.legend()
        plt.show()
        
# Task 1
def linear_regression_all_countries(country_data):
    """
    Fits a simple linear regression model (time -> cumulative cases)
    for each country in country_data.

    Returns:
        linear_models : dict {country: trained LinearRegression()}
        metrics_linear : dict {country: MSE value}
        preds_linear : dict {country: (t, y_pred)}
    """
    linear_models = {}
    metrics_linear = {}
    preds_linear = {}

    for country, (t, new_cases, cum) in country_data.items():
        # Prepare data
        X = t.reshape(-1, 1)
        y = cum

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predict
        y_pred = model.predict(X)

        # Store results
        linear_models[country] = model
        mse = mean_squared_error(y, y_pred)
        metrics_linear[country] = mse
        preds_linear[country] = (t, y_pred)

        # Plot
        plt.figure()
        plt.plot(t, y, "o", label="Data (cumulative)")
        plt.plot(t, y_pred, "-", label="Linear fit")
        plt.xlabel("Time [days]")
        plt.ylabel("Cumulative cases")
        plt.title(f"Task 1 — {country}: Linear regression\nMSE = {mse:.2f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return linear_models, metrics_linear, preds_linear

# Topic 2 Task 2
def polynomial_regression(country_data, poly_degree=3):
    """
    Fits a polynomial regression model for each country in country_data.
    Plots the results and returns dictionaries with models and MSE values.
    """
   

    poly_models = {}
    metrics_poly = {}

    for country, (t, new_cases, cum) in country_data.items():

        # Prepare data
        X = np.asarray(t).reshape(-1, 1)
        y = np.asarray(cum)

        # Build polynomial regression model
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=True)),
            ("linreg", LinearRegression())
        ])

        # Train
        model.fit(X, y)

        # Predict
        y_pred = model.predict(X)

        # Store model & MSE
        poly_models[country] = model
        mse = mean_squared_error(y, y_pred)
        metrics_poly[country] = mse

        # ----------- Plot -----------
        plt.figure()
        plt.plot(t, y, "o", label="Data (cumulative)")
        plt.plot(t, y_pred, "-", label=f"Poly deg {poly_degree}")
        plt.xlabel("Time [days]")
        plt.ylabel("Cumulative cases")
        plt.title(
            f"Task 2 — {country}: Polynomial regression (deg {poly_degree})\n"
            f"MSE = {mse:.2f}"
        )
        plt.grid(True)
        plt.legend()
        plt.show()

    return poly_models, metrics_poly

# Task 3

def train_mlp_all_countries(country_data, hidden=(32, 16), train_frac=0.8,
                            max_iter=10_000, random_state=0):
    """
    Train one MLP per country on (time -> cumulative cases),
    using the first `train_frac` of days for training and the rest for testing.
    """
    models = {}
    test_mse = {}
    preds = {}

    for country, (t, new_cases, cum) in country_data.items():
        t = np.asarray(t, float)
        y = np.asarray(cum, float)

        # Scale time to [0, 1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            X = (t - t_min) / (t_max - t_min)
        else:
            X = np.zeros_like(t)
        X = X.reshape(-1, 1)

        # Chronological split
        n = len(t)
        split = int(train_frac * n)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
        )
        mlp.fit(X_train, y_train)

        # Store results
        y_pred_all = mlp.predict(X)
        mse = mean_squared_error(y_test, mlp.predict(X_test))

        models[country] = mlp
        test_mse[country] = mse
        preds[country] = (t, y_pred_all, split)

    return models, test_mse, preds

def plot_mlp_country(country, t, cum, y_pred_all, split_idx, mse):
    plt.figure()
    plt.plot(t, cum, "o", label="Data")
    plt.plot(t, y_pred_all, "-", label="MLP prediction")
    plt.axvline(t[split_idx], linestyle="--", label="train/test split")
    plt.title(f"{country}: MLP regression (test MSE = {mse:.2f})")
    plt.xlabel("Time [days]")
    plt.ylabel("Cumulative confirmed cases")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
#Task 4 
def timeseries_to_supervised(data, lag=1):
    """
    data: 1D or 2D array (time series)
    lag : number of past time steps to include as input
    returns DataFrame with lag columns + current value as last column
    """
    df = DataFrame(data)
    # create columns: t-1, t-2, ..., t-lag
    cols = [df.shift(i) for i in range(lag, 0, -1)]
    cols.append(df)  # current value at time t
    agg = concat(cols, axis=1)
    agg.dropna(inplace=True)    # <-- drop the first 'lag' rows
    return agg

#Task 5
def lstm_all_countries(country_data, lookback=7, epochs=100):
    metrics_lstm = {}

    for country, (t, new_cases, cum) in country_data.items():
        series = np.asarray(cum, dtype="float32").reshape(-1, 1)

        # 1) supervised matrix
        supervised = timeseries_to_supervised(series, lag=lookback)
        values = supervised.values

        # 2) scale all columns to [0, 1]
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaled = scaler.fit_transform(values)

        # 3) chronological 80/20 split
        n_samples = scaled.shape[0]
        n_train = int(0.8 * n_samples)
        train, test = scaled[:n_train, :], scaled[n_train:, :]

        X_train, y_train = train[:, :-1], train[:, -1]
        X_test,  y_test  = test[:, :-1],  test[:, -1]

        # 4) reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test  = X_test.reshape((X_test.shape[0],  1, X_test.shape[1]))

        # 5) LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=1,
            verbose=0,
        )

        # 6) predictions on test set
        y_pred_scaled = model.predict(X_test, verbose=0)

        X_test_flat = X_test.reshape((X_test.shape[0], X_test.shape[2]))
        inv_y_pred = np.concatenate((X_test_flat, y_pred_scaled), axis=1)
        inv_y_pred = scaler.inverse_transform(inv_y_pred)[:, -1]

        inv_y_test = np.concatenate((X_test_flat, y_test.reshape(-1, 1)), axis=1)
        inv_y_test = scaler.inverse_transform(inv_y_test)[:, -1]

        mse_test = np.mean((inv_y_test - inv_y_pred) ** 2)
        metrics_lstm[country] = mse_test

        # 7) predict whole range
        X_all = scaled[:, :-1]
        X_all_lstm = X_all.reshape((X_all.shape[0], 1, X_all.shape[1]))
        y_all_scaled = model.predict(X_all_lstm, verbose=0)

        X_all_flat = X_all.reshape((X_all.shape[0], X_all.shape[1]))
        inv_y_all = np.concatenate((X_all_flat, y_all_scaled), axis=1)
        inv_y_all = scaler.inverse_transform(inv_y_all)[:, -1]

        # timeline
        t_seq = np.asarray(t[lookback:])
        t_train = t_seq[:n_train]
        t_test  = t_seq[n_train:]

        # plot
        plt.figure(figsize=(7, 5))
        plt.plot(t, series.flatten(), "o-", markersize=3, label="Data")
        plt.plot(t_seq, inv_y_all, "-", linewidth=2, label="LSTM prediction")
        plt.axvspan(t_train[0], t_train[-1], color="0.9", alpha=0.5, label="train window")
        plt.axvspan(t_test[0],  t_test[-1],  color="0.8", alpha=0.5, label="test window")
        plt.title(f"Task 4 – LSTM prediction for {country}\nMSE = {mse_test:.2f}")
        plt.xlabel("Time [days]")
        plt.ylabel("Cumulative confirmed cases")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return metrics_lstm

