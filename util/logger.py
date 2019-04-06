import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO


class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    # def image_summary(self,tag,images,step):
    #     pass

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a historgram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # drop first bin
        bin_edges = bin_edges[1:]

        # add bin edge and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)

        for c in counts:
            hist.bucket.append(c)

        # create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()