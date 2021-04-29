"""Show how to write a custom split action."""

from phy import IPlugin, connect
import numpy as np
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# def k_means(x,nclus):
#     """Cluster an array into two subclusters, using the K-means algorithm."""
#
#
#     return KMeans(n_clusters=nclus).fit_predict(x)


class ExampleCustomSplitFeaturePlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(
                submenu="Clustering",
                shortcut="s",
                prompt=True,
                n_args=5,
                # prompt_default=lambda: [139,1,133,1,2],
            )
            def kmeans(chan1,comp1,chan2,comp2,n_clusters):
                """Split using the K-means clustering: channel1 pc1 channel2 pc2 nclusters"""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.

                print("Selected channels = ",chan1,chan2)
                channel1 = np.where(controller.model.channel_mapping==chan1)[0]
                channel2 = np.where(controller.model.channel_mapping==chan2)[0]
                m1 = controller._get_features(cluster_ids[0], channel1, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2, load_all=True)

                a1 = m1.data[:, :, comp1 - 1]
                a2 = m2.data[:, :, comp2 - 1]

                a_comb = np.concatenate((a1, a2), axis=1)
                # a_comb = StandardScaler().fit_transform(a_comb)
                km = KMeans(n_clusters=int(n_clusters)).fit(a_comb)
                labels = km.labels_

                # We get the spike ids and the corresponding spike template amplitudes.
                # # NOTE: in this example, we only consider the first selected cluster.
                spike_ids = m1.spike_ids
                # y = bunchs[0].amplitudes

                # # We perform the clustering algorithm, which returns an integer for each
                # # subcluster.
                # labels = k_means(y.reshape((-1, 1)))
                assert spike_ids.shape == labels.shape

                # # We split according to the labels.
                controller.supervisor.actions.split(spike_ids, labels)

        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(
                submenu="Clustering",
                shortcut="y",
                prompt=True,
                n_args=5,
                # prompt_default=lambda: [139,1,133,1,2],
            )
            def spectral(chan1,comp1,chan2,comp2,n_clusters):
                """Split using the K-means clustering: channel1 pc1 channel2 pc2 nclusters"""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.
                print("Selected channels = ",chan1,chan2)
                channel1 = np.where(controller.model.channel_mapping==chan1)[0]
                channel2 = np.where(controller.model.channel_mapping==chan2)[0]
                m1 = controller._get_features(cluster_ids[0], channel1, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2, load_all=True)

                a1 = m1.data[:, :, comp1 - 1]
                a2 = m2.data[:, :, comp2 - 1]

                a_comb = np.concatenate((a1, a2), axis=1)
                km = SpectralClustering(n_clusters=int(n_clusters)).fit(a_comb)
                labels = km.labels_

                # We get the spike ids and the corresponding spike template amplitudes.
                # # NOTE: in this example, we only consider the first selected cluster.
                spike_ids = m1.spike_ids
                # y = bunchs[0].amplitudes

                # # We perform the clustering algorithm, which returns an integer for each
                # # subcluster.
                # labels = k_means(y.reshape((-1, 1)))
                assert spike_ids.shape == labels.shape

                # # We split according to the labels.
                controller.supervisor.actions.split(spike_ids, labels)

        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(
                submenu="Clustering",
                shortcut="z",
                prompt=True,
                n_args=5,
                # prompt_default=lambda: [139,1,133,1,2],
            )
            def agglomerative(chan1,comp1,chan2,comp2,n_clusters):
                """Split using the K-means clustering: channel1 pc1 channel2 pc2 nclusters"""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.

                print("Selected channels = ",chan1,chan2)
                channel1 = np.where(controller.model.channel_mapping==chan1)[0]
                channel2 = np.where(controller.model.channel_mapping==chan2)[0]
                m1 = controller._get_features(cluster_ids[0], channel1, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2, load_all=True)

                a1 = m1.data[:, :, comp1 - 1]
                a2 = m2.data[:, :, comp2 - 1]

                a_comb = np.concatenate((a1, a2), axis=1)
                km = AgglomerativeClustering(n_clusters=int(n_clusters)).fit(a_comb)
                labels = km.labels_

                # We get the spike ids and the corresponding spike template amplitudes.
                # # NOTE: in this example, we only consider the first selected cluster.
                spike_ids = m1.spike_ids
                # y = bunchs[0].amplitudes

                # # We perform the clustering algorithm, which returns an integer for each
                # # subcluster.
                # labels = k_means(y.reshape((-1, 1)))
                assert spike_ids.shape == labels.shape

                # # We split according to the labels.
                controller.supervisor.actions.split(spike_ids, labels)

        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(
                submenu="Clustering",
                shortcut="w",
                prompt=True,
                n_args=5,
                # prompt_default=lambda: [139,1,133,1,2],
            )
            def outlier(chan1,comp1,chan2,comp2,n_clusters):
                """Split using the K-means clustering: channel1 pc1 channel2 pc2 nclusters"""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.

                print("Selected channels = ",chan1,chan2)
                channel1 = np.where(controller.model.channel_mapping==chan1)[0]
                channel2 = np.where(controller.model.channel_mapping==chan2)[0]
                m1 = controller._get_features(cluster_ids[0], channel1, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2, load_all=True)

                a1 = m1.data[:, :, comp1 - 1]
                a2 = m2.data[:, :, comp2 - 1]

                a_comb = np.concatenate((a1, a2), axis=1)
                km = OneClassSVM(gamma='auto').fit_predict(a_comb)
                labels = km.labels_
                labels = np.where(labels>0,1,0)

                # We get the spike ids and the corresponding spike template amplitudes.
                # # NOTE: in this example, we only consider the first selected cluster.
                spike_ids = m1.spike_ids
                # y = bunchs[0].amplitudes

                # # We perform the clustering algorithm, which returns an integer for each
                # # subcluster.
                # labels = k_means(y.reshape((-1, 1)))
                assert spike_ids.shape == labels.shape

                # # We split according to the labels.
                controller.supervisor.actions.split(spike_ids, labels)
