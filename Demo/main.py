import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("data/data.csv")
print(data.head())
print(data.info())
data.dropna(inplace=True, subset=['Sell-to Customer No'], how='any')
data['Discount'] = data['Discount'].fillna(value=0)
print(data.info())


def extract_revenue_info(data):
    result = data[['Sell-to Customer No', 'Amount Including VAT']]
    result = result.groupby('Sell-to Customer No').sum()
    result.rename(columns={'Amount Including VAT': 'Revenue'}, inplace=True)
    return result


def extract_discount_info(data):
    result = data[['Sell-to Customer No', 'Discount']]
    # for i in result['Discount']:
    #       result['Discount'] = result(i).replace(',', '.')
    result['Discount'] = result['Discount'].apply(lambda x: 1 if x > 0 else 0)
    result = result.groupby('Sell-to Customer No').mean()
    return result


features = extract_revenue_info(data=data)
features = features.merge(extract_discount_info(data=data), left_index=True, right_index=True)

print(features)


def execute_K_means_automatic(features, K=10):
    cluster_values = list(range(2, K))
    silhouette_score_list = []
    best_silhouette_value = None
    best_model = None
    scaler = StandardScaler()
    scaler_features = scaler.fit_transform(features)
    for c in cluster_values:
        model = KMeans(n_clusters=c, init='k-means++', max_iter=500, random_state=42)
        model.fit(scaler_features)
        silhouette = silhouette_score(scaler_features, model.labels_)
        silhouette_score_list.append(silhouette)
        if best_silhouette_value is None or silhouette > best_silhouette_value:
            best_silhouette_value = silhouette
            best_model = model

    title_font = {'family': 'serif', 'color': 'red', 'weight': 'normal', 'size': 15}
    label_font = {'family': 'serif', 'color': 'black', 'weight': 'bold', 'size': 11}
    fig, ax = plt.subplots()
    ax.plot(range(2, K), silhouette_score_list, "bo--")
    ax.set_xlabel("n Cluster", fontdict=label_font)
    ax.set_ylabel("silhouette score", fontdict=label_font)
    ax.set_title("Gia tri Silhouette score", fontdict=title_font)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
    features['ClusterId'] = best_model.labels_


execute_K_means_automatic(features=features)
print(features)

THRES_DECIDE = 0.45


def get_percentile_value(data, percentiles):
    result = np.percentile(data, percentiles)
    return result


discount_percentiles_value = get_percentile_value(data=features['Discount'], percentiles=[25, 50, 75])
revenue_percentiles_value = get_percentile_value(data=features['Revenue'], percentiles=[25, 50, 75])


def get_percentile_code(data, percentiles_value):
    num_percentiles = len(percentiles_value)
    result = [0] * (num_percentiles + 1)
    if data >= percentiles_value[-1]:
        result[-1] = 1
    else:
        for i in range(0, num_percentiles):
            if data < percentiles_value[i]:
                result[i] = 1
                break
    return pd.Series(result)


features[['Discount Low', 'Discount Medium', 'Discount High', 'Discount very High']] = features['Discount'].apply(
    get_percentile_code,
    percentiles_value=discount_percentiles_value)

features[['Revenue Low', 'Revenue Medium', 'Revenue High', 'Revenue very High']] = features['Revenue'].apply(
    get_percentile_code,
    percentiles_value=revenue_percentiles_value)

for group in np.unique(features['ClusterId']):
    print("\n\nGROUP:", group)
    group_data = features[features['ClusterId'] == group]

    use_discount = np.mean(group_data[['Discount Low', 'Discount Medium', 'Discount High', 'Discount very High']])
    if use_discount.loc['Discount Low'] >= THRES_DECIDE:
        print("Discount Low")
    if use_discount.loc['Discount Medium'] >= THRES_DECIDE:
        print("Discount Medium")
    if use_discount.loc['Discount High'] >= THRES_DECIDE:
        print("Discount High")
    if use_discount.loc['Discount very High'] >= THRES_DECIDE:
        print("Discount very High")

    revenue = np.mean(group_data[['Revenue Low', 'Revenue Medium', 'Revenue High', 'Revenue very High']])
    if revenue.loc['Revenue Low'] >= THRES_DECIDE:
        print("Revenue Low")
    if revenue.loc['Revenue Medium'] >= THRES_DECIDE:
        print("Revenue Medium")
    if revenue.loc['Revenue High'] >= THRES_DECIDE:
        print("Revenue High")
    if revenue.loc['Revenue very High'] >= THRES_DECIDE:
        print("Revenue very High")
