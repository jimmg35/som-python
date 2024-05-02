from pyecharts import options as opts
from pyecharts.charts import Bar
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pyecharts.commons.utils import JsCode
from sklearn.metrics import cohen_kappa_score

def summary_cluster(cluster):
    grouped_cluster = cluster.groupby('cluster')

    xaxis = []
    yaxis = []
    for group_name, group_data in grouped_cluster:
        xaxis.append(int(group_name))
        yaxis.append(len(group_data))

    # 將xaxis和yaxis打包成一個tuple的列表
    xy_pairs = list(zip(xaxis, yaxis))

    # 使用sorted函數對這個列表進行排序，排序的依據是每個tuple的第一個元素（也就是xaxis的值）
    sorted_pairs = sorted(xy_pairs, key=lambda pair: pair[1])

    # 使用zip函數將排序後的列表解包成兩個列表
    xaxis_sorted, yaxis_sorted = zip(*sorted_pairs)

    chart = Bar()

    chart.add_xaxis(xaxis_sorted)
    chart.add_yaxis("像素數", yaxis_sorted)
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title="Cluster Description"),
    )
    return chart
    

def one_hot_encode_clusters(cluster):
    unique_cluster = np.unique(cluster['cluster'])
    one_hot_encoded_clusters = []
    for cluster_id in unique_cluster:
        cluster_copy = cluster.copy()
        cluster_copy['cluster'] = np.where(cluster_copy['cluster'] == cluster_id, 1, 0)
        one_hot_encoded_clusters.append(cluster_copy)
    return one_hot_encoded_clusters


def random_forest_train(one_hot_encoded_clusters, cluster_index):
    one_hot_encoded_cluster = one_hot_encoded_clusters[cluster_index]
    selected_features = [feature for feature in one_hot_encoded_cluster.columns if feature not in ['id', 'cluster']]

    # 選取特定的欄位作為特徵
    X = one_hot_encoded_cluster[selected_features]  
    y = one_hot_encoded_cluster['cluster']

    # 將資料集分為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 初始化隨機森林分類器
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # 在訓練集上訓練模型
    rf_classifier.fit(X_train, y_train)

    # 在測試集上進行預測
    y_pred = rf_classifier.predict(X_test)

    # 評估模型性能
    accuracy = np.mean(y_pred == y_test)
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"Cluster {cluster_index} | Accuracy: {accuracy} | Kappa: {kappa}")

    
    # 獲取特徵重要性分數
    feature_importances = rf_classifier.feature_importances_

    # # 將特徵重要性分數與特徵名稱對應起來
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    return [feature_importance_df, kappa]


def render_importance_rank(feature_importance_df, cluster_index):
    
    c = (
        Bar()
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"Top 10 Important Features of Cluster {cluster_index}"),
        )
        .add_xaxis(list(feature_importance_df['Feature']))
        .add_yaxis('Importance', list(feature_importance_df['Importance']))
        .set_series_opts(
            label_opts=opts.LabelOpts(
                position="top",
                formatter=JsCode(
                    """
                    function(params) {
                        return params.value.toFixed(2);
                    }
                    """
                ),
            )
        )
    )
    return c

    #bar = Bar()
    #bar.set_global_opts(
    #    title_opts=opts.TitleOpts(title=f"Top 10 Important Features of Cluster {cluster_index}"),
    #)
    #bar.add_xaxis(list(feature_importance_df['Feature']))
    #bar.add_yaxis('Importance', list(feature_importance_df['Importance']))

    #bar.set_series_opts(
    #        label_opts=opts.LabelOpts(
    #            position="top",
    #            formatter=JsCode(
    #                """
    #                function(params) {
    #                    return params.value.toFixed(2);
    #                }
    #                """
    #            ),
    #        )
    #    )
    #return bar