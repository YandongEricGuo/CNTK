import easydict
from utils.nms_wrapper import apply_nms_to_single_image_results

def train_object_detector(cfg):
    """
    Trains an object detector as specified in the configuration
    :param cfg: the configuration
    :return: the eval model of the trained detector
    """

    detector_name = _get_detector_name(cfg)
    eval_model = None
    if detector_name is not None:
        print("training {}".format(detector_name))
        if detector_name == 'FasterRCNN':
            from FasterRCNN.FasterRCNN import prepare, train_faster_rcnn
            prepare(cfg, use_arg_parser=False)
            eval_model = train_faster_rcnn(cfg)
        else:
            print('Unknown detector: {}'.format(detector_name))

    return eval_model

def evaluate_test_set(model, cfg):
    """
    Evaluates the given model on the test set as specified in the configuration
    :param model: the model
    :param cfg: the configuration
    :return: AP (average precision) per class
    """

    detector_name = _get_detector_name(cfg)
    aps = None
    if detector_name is not None:
        print("evaluating {}".format(detector_name))
        if detector_name == 'FasterRCNN':
            from FasterRCNN.FasterRCNN import evaluate_faster_rcnn
            aps = evaluate_faster_rcnn(model, cfg)
        else:
            print('Unknown detector: {}'.format(detector_name))

    return aps

def evaluate_single_image(model, img_path, cfg):
    """
    Computes detection results for the given model on the provided image
    :param model: the model
    :param img_path: the path to the image
    :param cfg: the configuration
    :return:
        regressed_rois - the predicted bounding boxes
        cls_probs - class probabilities per bounding box
    """

    detector_name = _get_detector_name(cfg)
    regressed_rois = None
    cls_probs = None
    if detector_name is not None:
        print("detecting objects in image {}".format(img_path))
        if detector_name == 'FasterRCNN':
            from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
            evaluator = FasterRCNN_Evaluator(model, cfg)
            regressed_rois, cls_probs = evaluator.process_image(img_path)

            if False:
                import time
                start = int(time.time())
                for i in range(100):
                    regressed_rois, cls_probs = evaluator.process_image(img_path)
                total = int(time.time()) - start
                print("time for 100 evals: {}".format(total))

        else:
            print('Unknown detector: {}'.format(detector_name))

    return regressed_rois, cls_probs

def filter_results(regressed_rois, cls_probs, cfg):
    """
    Filters the provided results by performing NMS (non maximum suppression)
    :param regressed_rois: the predicted bounding boxes
    :param cls_probs: class probabilities per bounding box
    :param cfg: the configuration
    :return:
        bboxes - the filtered list of bounding boxes
        labels - the single class label per bounding box
        scores - the probability for the assigned class label per bounding box
    """

    labels = cls_probs.argmax(axis=1)
    scores = cls_probs.max(axis=1)
    nmsKeepIndices = apply_nms_to_single_image_results(
                        regressed_rois, labels, scores,
                        use_gpu_nms=cfg.USE_GPU_NMS,
                        device_id=cfg.GPU_ID,
                        nms_threshold=cfg["CNTK"].RESULTS_NMS_THRESHOLD,
                        conf_threshold=cfg["CNTK"].RESULTS_NMS_CONF_THRESHOLD)

    filtered_bboxes = regressed_rois[nmsKeepIndices]
    filtered_labels = labels[nmsKeepIndices]
    filtered_scores = scores[nmsKeepIndices]

    return filtered_bboxes, filtered_labels, filtered_scores
    
def visualize_results(img_path, bboxes, labels, scores, cfg, store_to_path=None):
    """
    Renders the detection results (bboxes and labels) onto the image.
    :param img_path: the path to the image
    :param bboxes: the predicted bounding boxes
    :param labels: the single class label per bounding box
    :param scores: the probability for the assigned class label per bounding box
    :param cfg: the configuration
    :param store_to_path: optional: a path where to store the rendered image.
                          If set to 'None' the image will be displayed on screen.
    :return:
    """

    from matplotlib.pyplot import imsave, imshow, show
    from utils.plot_helpers import visualize_detections
    img = visualize_detections(img_path, bboxes, labels, scores,
                               cfg["CNTK"].IMAGE_WIDTH, cfg["CNTK"].IMAGE_HEIGHT,
                               classes = cfg["CNTK"].CLASSES,
                               draw_negative_rois = cfg["CNTK"].DRAW_NEGATIVE_ROIS)

    if store_to_path is not None:
        imsave(store_to_path, img)
    else:
        imshow(img)
        show()

def _get_detector_name(cfg):
    try:
        detector = cfg['DETECTOR']
    except:
        print("Please specify a 'DETECTOR' in your configuration.")
        detector = None
    return detector
