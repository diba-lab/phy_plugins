"""Show how to write a custom split action."""

from phy import IPlugin, connect
import numpy as np
from sklearn.cluster import KMeans, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def _uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


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
            def kmeans(chan1, comp1, chan2, comp2, n_clusters):
                """Split using the K-means clustering: channel1 pc1 channel2 pc2 nclusters"""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.

                print("Selected channels = ", chan1, chan2)
                channel1 = np.where(controller.model.channel_mapping == chan1)[0]
                channel2 = np.where(controller.model.channel_mapping == chan2)[0]
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
                shortcut="z",
                prompt=True,
                n_args=4,
                # prompt_default=lambda: [139,1,133,1,2],
            )
            def mean_shift(chan1, comp1, chan2, comp2):
                """Split using mean_shift algorithm: channel1 pc1 channel2 pc2 nclusters"""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.

                print("Selected channels = ", chan1, chan2)
                channel1 = np.where(controller.model.channel_mapping == chan1)[0]
                channel2 = np.where(controller.model.channel_mapping == chan2)[0]
                m1 = controller._get_features(cluster_ids[0], channel1, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2, load_all=True)

                a1 = m1.data[:, :, comp1 - 1]
                a2 = m2.data[:, :, comp2 - 1]

                a_comb = np.concatenate((a1, a2), axis=1)
                labels = MeanShift(bandwidth=2).fit_predict(a_comb)

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
                n_args=4,
                # prompt_default=lambda: [139,1,133,1,2],
            )
            def outlier(chan1, comp1, chan2, comp2):
                """Outlier params: channel1 pc1 channel2 pc2"""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.

                print("Selected channels = ", chan1, chan2)
                channel1 = np.where(controller.model.channel_mapping == chan1)[0]
                channel2 = np.where(controller.model.channel_mapping == chan2)[0]
                m1 = controller._get_features(cluster_ids[0], channel1, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2, load_all=True)

                a1 = m1.data[:, :, comp1 - 1]
                a2 = m2.data[:, :, comp2 - 1]

                a_comb = np.concatenate((a1, a2), axis=1)
                labels = LocalOutlierFactor(n_neighbors=20).fit_predict(a_comb)
                labels = np.where(labels == 1, 1, 0)

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


class ReclusterGaussian(IPlugin):
    def attach_to_controller(self, controller):
        # self.manual_selection = controller.selection.channel_id

        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(
                shortcut="b",
                prompt=True,
                n_args=3,
                # prompt_default=lambda: [139,1,133,1,2],
            )
            def gaussian(chan1, chan2, n_clusters):
                """Split using gaussian algorithm: channel1 channel2 nclusters"""
                # manual_selection = []
                # try:
                #     manual_selection.append(controller.selection.channel_id)
                # except:
                #     selected_cluster = controller.supervisor.selected
                #     best_chans = controller.get_best_channels(selected_cluster[0])
                #     manual_selection.append(best_chans[:2])

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected
                # selected_channels = controller.get_best_channels(cluster_ids[0])

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.
                # chan1, chan2 = self.channel_ids[:2]
                print("Selected channels = ", chan1, chan2)
                # chan1, chan2 = selected_channels[:2]
                channel1 = np.where(controller.model.channel_mapping == chan1)[0]
                channel2 = np.where(controller.model.channel_mapping == chan2)[0]
                m1 = controller._get_features(cluster_ids[0], channel1, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2, load_all=True)

                a1 = m1.data[:, :, :].squeeze()
                a2 = m2.data[:, :, :].squeeze()

                a_comb = np.concatenate((a1, a2), axis=1)

                labels = GaussianMixture(n_components=n_clusters).fit_predict(a_comb)
                # labels = model.predict(a_comb)

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
