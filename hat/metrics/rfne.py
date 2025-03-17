import numpy as np

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_rfne(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate RFNE (Relative Frobenius Norm Error).
    
    RFNE is a common metric for evaluating climate model results,
    calculated as: ||img - img2||_F / ||img2||_F
    where ||.||_F denotes the Frobenius norm.

    Args:
        img (ndarray): Predicted images.
        img2 (ndarray): Ground truth images.
        crop_border (int): Cropped pixels in each edge of an image.
            These pixels are not involved in the RFNE calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
            For climate data, this should generally be False.

    Returns:
        float: rfne result (lower is better).
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    
    # Reorder image if necessary
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Crop borders if specified
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to Y channel if specified (typically not needed for climate data)
    if test_y_channel:
        raise NotImplementedError("Y channel testing not implemented for RFNE as it's not typical for climate data")
    
    # Calculate Frobenius norm of the difference
    norm_diff = np.linalg.norm(img - img2)
    # Calculate Frobenius norm of the ground truth
    norm_gt = np.linalg.norm(img2)
    
    # Avoid division by zero
    if norm_gt == 0:
        return float('inf')
    
    # Return the relative error
    return norm_diff / norm_gt 