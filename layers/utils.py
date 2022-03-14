#Adapted from : https://gist.github.com/bbrighttaer/207dc03b178bbd0fef8d1c0c1390d4be

import torch


def segment_sum(data, segment_ids,device='cpu'):
    """
    Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    :param data: A pytorch tensor of the data for segmented summation.
    :param segment_ids: A 1-D tensor containing the indices for the segmentation.
    :return: a tensor of the same type as data containing the results of the segmented summation.
    """
    #Note: Eric fix this on 14/03/2022
    # if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
    #     raise AssertionError("elements of segment_ids must be sorted")

    # if len(segment_ids.shape) != 1:
    #     raise AssertionError("segment_ids have be a 1-D tensor")
    #
    # if data.shape[0] != segment_ids.shape[0]:
    #     raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    # num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, len(torch.unique(segment_ids)),device=device)


def unsorted_segment_sum(data, segment_ids, num_segments,device='cpu'):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape,device=device).scatter_add(0, segment_ids, data)
    # tensor = tensor.type(data.dtype)
    return tensor