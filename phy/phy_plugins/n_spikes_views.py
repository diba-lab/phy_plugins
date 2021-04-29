"""Show how to change the number of displayed spikes in each view."""

from phy import IPlugin


class ExampleNspikesViewsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Feel free to keep below just the values you need to change."""
        controller.n_spikes_waveforms = 100
        controller.batch_size_waveforms = 1000
        controller.n_spikes_features = 30000
        controller.n_spikes_features_background = 1
        controller.n_spikes_amplitudes = 5000
        controller.n_spikes_correlograms = 100000
        # Number of "best" channels kept for displaying the waveforms.
        controller.model.n_closest_channels = 12

        # The best channels are selected among the N closest to the best (peak) channel if their
        # mean amplitude is greater than this fraction of the peak amplitude on the best channel.
        # If zero, just the N closest channels are kept as the best channels.
        controller.model.amplitude_threshold = 0
