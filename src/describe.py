from pyecharts import options as opts
from pyecharts.charts import Bar
import numpy as np


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