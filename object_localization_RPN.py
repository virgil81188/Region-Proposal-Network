import os
#disable tensorflow warnings logging as described here: https://stackoverflow.com/questions/43134753/tensorflow-wasnt-compiled-to-use-sse-etc-instructions-but-these-are-availab
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import math
import cv2
import enum
import time
import json
from typing import List, Tuple

###################### helpers ######################

def image_resize(image, new_size):
    '''
        image - opencv image
        new_size - shape of new image in (height, width, channels) or in (height, width) format
    '''
    interpolation_method = cv2.INTER_AREA
    if (new_size[0] > image.shape[0] and new_size[1] > image.shape[1]):
            interpolation_method = cv2.INTER_LANCZOS4
    return cv2.resize(image,(new_size[1], new_size[0]), interpolation = interpolation_method)

def IoU_tensorial(A,B):
    '''
        Computes the Intersection over Union (IoU) of the rectangles in A vs those in B
        A - tensor containing rectangles in XYWH format
        B - tensor containing rectangles in XYWH format
        returns a tensor IoU, containing the IoU of each rectangle pair (a,b), where a is in A and b is in B
    '''
    A = np.copy(A)
    B = np.copy(B)
    A[:,2] = A[:,0] + A[:,2]
    A[:,3] = A[:,1] + A[:,3]

    B[:,2] = B[:,0] + B[:,2]
    B[:,3] = B[:,1] + B[:,3]

    nrA = A.shape[0] # number of rectangles in A
    nrB = B.shape[0] # number of rectangles in B
    eA = np.repeat(A[None, :, :], nrB, 0)
    eB = np.repeat(B[None, :, :], nrA, 0)
    eB = np.transpose(eB, [1, 0, 2]) # align the axes to match those of eA
    # we now have two tensors eA and eB so that the first dimension chooses a box in B, the second dimension chooses a box in A, the third dimension chooses box attribute
    # split eA and eB into halfs and perform max and min
    eAshape = eA[:,:,0:2].shape
    ul = np.maximum(eA[:,:,0:2].ravel(), eB[:,:,0:2].ravel()).reshape(eAshape) #upper left corner of intersection rectangle
    br = np.minimum(eA[:,:,2:4].ravel(), eB[:,:,2:4].ravel()).reshape(eAshape) #bottom right corner of intersection rectangle
    w = np.clip(br[:,:,0] - ul[:,:,0], 0, np.Infinity) #width of the intersection rectangle
    h = np.clip(br[:,:,1] - ul[:,:,1], 0, np.Infinity) #height of the intersection rectangle

    I = w * h # the intersection areas
    U = (eA[:,:,2] - eA[:,:,0]) * (eA[:,:,3] - eA[:,:,1]) + (eB[:,:,2] - eB[:,:,0]) * (eB[:,:,3] - eB[:,:,1]) - I # the union areas
    IoU = I / U # the IoU scores of all the rectangle pairs in A and B
    return np.transpose(IoU)

###################### dataset ######################

def display_training_sample(path):
    '''
    displays a training sample image with ground truth rectangles overlayed
    '''
    dir,filename = os.path.split(path)
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        fnamewithoutextension = os.path.splitext(filename)[0]
        input_file = open(os.path.join(dir, fnamewithoutextension+".bboxes.tsv"), "r")
        image = cv2.imread(path)
        boxes = []
        for line in input_file:
            line = line.strip()
            ss = line.split('\t')
            x = int(ss[0])
            y = int(ss[1])
            w = int(ss[2])
            h = int(ss[3])
            label = ss[4]
            boxes.append([x, y, w, h, label])
        for box in boxes:
            cv2.rectangle(image, (int(np.clip(box[0], 0, image.shape[0])), int(np.clip(box[1], 0, image.shape[1]))), (int(np.clip(box[0]+box[2], 0, image.shape[0])), int(np.clip(box[1] + box[3], 0, image.shape[1]))), (0,255,0), 2)
        cv2.imshow("ground truth: " + path, image)
        cv2.waitkey(0)

def get_training_sample_image_annotated(path):
    '''
    returns the image of a training sample with the ground truth annotations overlayed
    '''
    dir,filename = os.path.split(path)
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        fnamewithoutextension = os.path.splitext(filename)[0]
        input_file = open(os.path.join(dir, fnamewithoutextension+".bboxes.tsv"), "r")
        image = cv2.imread(path)
        boxes = []
        for line in input_file:
            line = line.strip()
            ss = line.split('\t')
            x = int(ss[0])
            y = int(ss[1])
            w = int(ss[2])
            h = int(ss[3])
            label = ss[4]
            if (w > 0 and h > 0):
                boxes.append([x, y, w, h, label])
        for box in boxes:
            cv2.rectangle(image, (int(np.clip(box[0], 0, image.shape[0])), int(np.clip(box[1], 0, image.shape[1]))), (int(np.clip(box[0]+box[2], 0, image.shape[0])), int(np.clip(box[1] + box[3], 0, image.shape[1]))), (0,255,0), 2)
        return image
    return None

class LearningSets(enum.Enum):
    fullset, trainset, validationset, testset = range(4)

class DetectionDataset:
    def __init__(self, images_path, image_height, image_width, channels, keep_aspect_ratio):
        self.images_path = images_path
        self.loaded = False
        self.x_val = None
        self.y_val =  None        
        self.train_set = None
        self.test_set = None
        self.image_height, self.image_width, self.channels,self.keep_aspect_ratio = image_height, image_width, channels, keep_aspect_ratio
        self._classesdict = None
        self.start_produce_records = None
        self.data_dict = {}
        self.train_set = []
        self.validation_set = []
        self.test_set_fraction = 0.2
        self.classes = []

    def load(self, images_have_same_size : bool = False):
        if self.loaded  == False:
            self.loaded = True
            got_size = False
            image_size = (self.image_height, self.image_width)
            for root, dirnames, filenames in os.walk(self.images_path):
                for filename in filenames:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                        filepath = os.path.join(root, filename)
                        file_name_without_extension = os.path.splitext(filename)[0]
                        self.data_dict[file_name_without_extension] = {'path': filepath,'boxes' : None}
                        input_file = open(os.path.join(root, file_name_without_extension+".bboxes.tsv"), "r")
                        if not got_size:
                            im = cv2.imread(filepath)
                            image_size = im.shape
                        if images_have_same_size:
                            got_size = True
                        wr = self.image_width / image_size[1]
                        hr = self.image_height / image_size[0]
                        boxes = []
                        for line in input_file:
                            line = line.strip()
                            ss = line.split('\t')
                            x = int(ss[0])
                            y = int(ss[1])
                            w = int(ss[2])
                            h = int(ss[3])
                            label = ss[4]
                            if label not in self.classes:
                                self.classes.append(label)
                            if (w > 0 and h > 0):
                                boxes.append([x * wr, y * hr, w * wr, h * hr, self.classes.index(label)])
                        #boxes = boxes[0:self.max_objects] #don't use more than self.max_objects objects
                        self.data_dict[file_name_without_extension]['boxes'] = boxes
                break
            all_keys = [k for k in self.data_dict.keys()]
            # sample current class directory and retreive testsetfraction fraction of the files
            self.validation_set = list(np.random.choice(all_keys , int(self.test_set_fraction * len(all_keys)), replace = False))
            # place the rest of the elements in the train set
            self.train_set = list(set(all_keys) - set(self.validation_set))
            assert len(self.validation_set) + len(self.train_set) == len(all_keys)

    def get_random_example(self):
        self.load()
        all_examples = [ v for v in self.data_dict.values() ]
        random_idx = np.random.choice(len(all_examples), 1, replace = False)
        im = cv2.imread(all_examples[random_idx]['path'])
        if (im is None):
            return None
        # resize the image to fit into memory
        im = image_resize(im, (self.image_height, self.image_width))
        return (np.asarray(im), np.asarray(all_examples[random_idx]['boxes']), all_examples[random_idx])

    def get_batch(self, batch_size : int, learning_set : LearningSets):
        self.load()
        l = {
        LearningSets.fullset : [v for v in self.data_dict.keys()],
        LearningSets.trainset : self.train_set,
        LearningSets.validationset: self.validation_set,
        LearningSets.testset : self.test_set
        }.get(learning_set, [])

        randomidx = np.random.choice(len(l), batch_size, replace = False)
        batch = []
        for i in randomidx:
            example = self.data_dict[l[i]]
            #print("image:"+all_examples[i]['path'])
            im = cv2.imread(example['path'])
            if (im is None):
                continue
            # resize the image to fit into memory
            im = image_resize(im, (self.image_height, self.image_width))
            batch.append((np.asarray(im), np.asarray(example['boxes'])))
        return np.asarray(batch)

    @property
    def number_of_train_samples(self):
        return len(self.data_dict)

    @property
    def classes_dict(self):
        if self._classesdict is None:
            for i, cls in enumerate(classes):
                self._classes_dict[cls] = i
        return self._classesdict

    def get_validation_data(self):
        self.load()
        return self.x_val, self.y_val


###################### layers ##########################

def conv2d(x, W, conv_bias ,stride = 1, padd_val = 'SAME', name:str = None):
    ret = tf.nn.conv2d(x, W, strides = [1,stride,stride,1], padding = padd_val)
    ret = tf.add(ret,conv_bias, name)
    return ret


def conv2d_relu(x, W, conv_bias ,stride = 1, padd_val = 'SAME', name:str = None):
    '''
    Convolution step -> ReLU
    '''
    ret = conv2d(x, W, conv_bias ,stride, padd_val)
    ret = tf.nn.relu(ret, name)
    return ret

def conv2d_batchnorm_relu(x, W, conv_bias ,stride = 1, padd_val = 'SAME', is_training = False, name = None):
    '''
    Convolution step -> Batch normalization -> ReLU
    '''
    ret = conv2d(x,W, conv_bias ,stride, padd_val)
    ret = tf.contrib.layers.batch_norm(ret, decay=0.9, is_training = is_training, scale = True, updates_collections=None)
    ret = tf.nn.relu(ret, name = name)
    return ret

def maxpool2d(x, size, stride,padd_val = 'SAME', name = None):
    return tf.nn.max_pool(x, ksize=[1,size,size,1], strides=[1,stride,stride,1], padding=padd_val, name = name)

def max_pool_with_fixed_output_size(X, desired_size_HW : Tuple[int, int], name : str = None):
    """
    X - 4D input tensor of shape = (batches, height, width, channels)
    desired_size_HW - 1D tensor giving the height and width of the output tensor

    Divides the input tensor along the axis=(1,2) with a grid of shape desired_size_HW,
    resulting in a matrix of pools, then takes the maximum over the axis=(1,2) in each pool,
    then combines the results of each pool in a single tensor of shape (X.shape[0], desired_size_HW[0], desired_size_HW[1], X.shape[3])

    Note: Input tensor should have height >= desired height and width >= desired width
    """
    inputShape = tf.shape(X)
    # compute pool size
    cellSize = [inputShape[1] // desired_size_HW[0], inputShape[2] // desired_size_HW[1]]
    outputs = []
    for line in range(desired_size_HW[0]):
        # compute kernel size on the "height" axis
        h = cellSize[0]
        if (line == desired_size_HW[0]-1):  # for the last pool, take all remaining lines of the input tensor
            h = -1
        for col in range(desired_size_HW[1]):
            x = col * cellSize[1]
            y = line * cellSize[0]
            # compute kernel size on the "width" axis
            w = cellSize[1]
            if (col == desired_size_HW[1] - 1): # for the last pool column, take all remaining cols of the input tensor
                w = -1
            cropShape = [-1, h, w, -1] # create pool
            cropPosition = [0, y, x, 0] # position pool
            crop = tf.slice(X, cropPosition, cropShape) # extract pool values from input tensor
            max = tf.reduce_max(crop, axis = [1,2]) # compute max for the current pool
            outputs.append(max) # add the current pool to the final output
    # concatenate and reshape pools to form a tensor with the given desired size
    #return tf.reshape(tf.concat(outputs, axis=1), shape=[-1, desired_size_HW[0], desired_size_HW[1], inputShape[3]], name = name)
    return tf.reshape(tf.concat(outputs, axis=1), shape=[-1, desired_size_HW[0], desired_size_HW[1], X.shape[3].value], name = name)

def SPPLayer(X, level_sizes : List[int], name : str = None):
    """
    Spatial Pyrmaid Pooling
    Technical paper: https://arxiv.org/pdf/1406.4729.pdf. Implementation differs a little from the paper.

    X - 4D input tensor of shape = (batches, height, width, channels)
    levelSizes - list of sizes for each pyramid level

    For each level i of the pyramid, it computes the MaxPoolWithFixedOutputSize(X, [levelSizes[i], levelSizes[i]])
    then it reshapes it (batch, channels * levelSizes[i]^2) and concatenates them into one 2D tensor of shape(batches, channels * (sum(levelSizes[i]^2)))
    """
    nrChannels = tf.shape(X)[3]
    lvl = X
    outputs = []
    for lvlSize in level_sizes:
        # compute the current level of the pyramid as a 2-D tensor
        lvl = tf.reshape(max_pool_with_fixed_output_size(X, [lvlSize, lvlSize]), shape=[-1, lvlSize * lvlSize * nrChannels])
        outputs.append(lvl)
    # form a single 2-D tensor with all the levels concatenated on one line, each line corresponding to a sample in the batch
    return tf.concat(outputs, axis=1, name = name)

def rpn_generate_anchors(inputSize : Tuple[int, int], convSize : Tuple[int, int], ratios : List[float], scales : List[float]):
    '''
        Generates a tensor with anchors for the Region Proposal Network (RPN) as described here: https://arxiv.org/pdf/1506.01497.pdf
        inputSize = (width, height) of the input image
        convSize = (width, height) of the feature map
        ratios = list of aspect ratios to use computed as width/height. e.g.: [0.5, 1, 2]
        scales = list of scales to use for the anchors. e.g.: [16, 32, 64]
        
        Returns a tensor with size (height, width, len(ratios)*len(scales), 4) meaning (convSize height, convSize width, number of anchors per feature map location, 
    '''
    k = len(ratios) * len(scales)
    A = np.zeros((convSize[1], convSize[0], k, 4))
    anchors = []
    for s in scales:
        Area = s*s
        for r in ratios:
            h = math.sqrt(Area / r)
            w = Area / h
            anchors.append((round(w),round(h)))

    for i in range(convSize[1]):
        for j in range(convSize[0]):
            x = round((j + 0.5) / convSize[0] * inputSize[0])
            y = round((i + 0.5) / convSize[1] * inputSize[1])
            for idx,a in enumerate(anchors):
                A[i,j,idx,:] = [x,y,a[0],a[1]]
    return A
         
def rpn_get_batch(image, GT, conv_size : Tuple[int, int], ratios : List[float], scales : List[float], minibatch_size : int = 128, positives_threshold : float = 0.5, negatives_threshold : float = 0.2, repeat_positives : bool = True):
        '''
            Generates a minibatch for the RPN training loss function
            image - input image as a numpy array
            GT - tensor of shape (-1,4) containing the ground truth rectangles given in the format: (x,y,w,h)<=>(upperleft,size)
            conv_size = (width, height) of the feature map
            ratios = list of aspect ratios to use computed as width/height. e.g.: [0.5, 1, 2]
            scales = list of scales to use for the anchors. e.g.: [16, 32, 64]

            Returns:
            return minibatch_indices, A_scores_one_hot, ground_truth_anchors_list
            minibatch_indices - the indices of the chosen anchors in the anchors list
            A_scores_one_hot - the scores of the ground truth anchors one hot encoded
            ground_truth_anchors_list - the ground truth achors as a list
        '''
        A = rpn_generate_anchors((image.shape[0], image.shape[1]), conv_size, ratios, scales)
        
        A_list = np.reshape(A, (-1,4))
        A_list_XYXY = A_list.copy()
        A_list_XYXY[:, 0] = A_list[:,0] - A_list[:,2] / 2
        A_list_XYXY[:, 1] = A_list[:,1] - A_list[:,3] / 2
        A_list_XYXY[:, 2] = A_list[:,0] + A_list[:,2] / 2
        A_list_XYXY[:, 3] = A_list[:,1] + A_list[:,3] / 2
        # create a GT rectangles with (x1,y1, x2,y2)<=>(upper left,bottom right) representation from GT rectangles with (x,y,w,h)<=>(upperleft,size) representation
        # needed for the IoU computation
        GT_XYXY = GT.copy() 
        GT_XYXY[:,2] += GT_XYXY[:,0]
        GT_XYXY[:,3] += GT_XYXY[:,1]
        IoU = IoU_tensorial(A_list_XYXY, GT_XYXY)
        idx_of_GT = np.argmax(IoU, axis = 1)
        A_scores_list = np.max(IoU, axis = 1)
        safety_object = np.argmax(A_scores_list)

        A_scores_list[A_scores_list>positives_threshold] = 1
        A_scores_list[A_scores_list<negatives_threshold] = 0
        A_scores_list[np.logical_and(A_scores_list>0, A_scores_list<1)] = 0.5

        object_indices = np.arange(len(A_scores_list))[A_scores_list > 0.99]
        non_object_indices = np.arange(len(A_scores_list))[A_scores_list < 0.1]

        # include at least one object
        if (len(object_indices) == 0):
            object_indices = np.array([safety_object])
            A_scores_list[safety_object] = 1
            idx = np.argwhere(non_object_indices == safety_object)
            non_object_indices = np.delete(non_object_indices, idx)

        assert(len(object_indices) > 0)

        if repeat_positives:
            chosen_idx_obj = np.random.choice(object_indices, int(minibatch_size/2),replace = True)
            assert(len(chosen_idx_obj) + len(non_object_indices) >= minibatch_size)
        else:
            assert(len(object_indices) + len(non_object_indices) >= minibatch_size)
            chosen_idx_obj = np.random.choice(object_indices, min(len(object_indices), int(minibatch_size/2)),replace = False)
        chosen_idx_non_obj = np.random.choice(non_object_indices, minibatch_size - len(chosen_idx_obj), replace = False)
        
        minibatch_indices = np.concatenate((chosen_idx_obj, chosen_idx_non_obj))

        # minibatch only
        ground_truth_anchors_list = np.zeros((len(minibatch_indices), 4))
        ground_truth_anchors_list[:,0] = (GT[idx_of_GT[minibatch_indices],0] - A_list[minibatch_indices,0]) / A_list[minibatch_indices,2]
        ground_truth_anchors_list[:,1] = (GT[idx_of_GT[minibatch_indices],1] - A_list[minibatch_indices,1]) / A_list[minibatch_indices,3]
        ground_truth_anchors_list[:,2] = np.log(GT[idx_of_GT[minibatch_indices],2]/A_list[minibatch_indices,2])
        ground_truth_anchors_list[:,3] = np.log(GT[idx_of_GT[minibatch_indices],3]/A_list[minibatch_indices,3])

        a = A_scores_list[minibatch_indices]
        A_scores_one_hot = np.zeros((a.shape[0], 2))
        A_scores_one_hot[np.arange(a.shape[0]), a.astype(int)] = 1
        return minibatch_indices, A_scores_one_hot, ground_truth_anchors_list

def rpn_layer(x, input_size : Tuple[int, int], channels : int = 256, scales : List[float] = [32, 64, 128], ratios : List[float] = [0.5, 1, 2], _confidence_threshold : float = 0.75 ,is_training = False, name : str = None):
    '''
        Region proposal network layer (implemented as described in https://arxiv.org/pdf/1506.01497.pdf )
        
        x - base input tensor
        input_size - size in pixels of the initial image size that entered the network, used compute the anchors
        channels - rpn feature channels
        scales - scales of the anchors
        ratios - ratios of the anchors
        _confidence_threshold - the minimum confidence that a box contains an object to be selected for box prediction
        is_training - tells whether the network is in training mode

        The function builds the rpn layer with the given parameters.
        It also defines a scalar (float) input used for prediction named: "confidenceThreshold"
        Returns 4 tensors: cls, reg, boxConfidences, prediction
        
        cls - tensor of shape (x.shape[0], x.shape[1], x.shape[2], 2 * len(scales) * len(ratios))
              contains pairs of logits giving the (noObject, Object) scores for all the proposals
        reg - tensor of shape (x.shape[0], x.shape[1], x.shape[2], 4 * len(scales) * len(ratios))
              contains tuples giving the (tx, ty, tw, th) parametrized deltas of all the proposals
        box_confidences - tensor of shape (-1) giving the confidence score for the proposals above the give confidence threshold
        prediction - tensor of shape (-1, 4) giving the coordinates in (left, top, width, height) format of the propsals that hav confidence above the confidence threshold
    '''

    in_channels = x.shape[3].value
    rpn_features_channels = channels
    nr_anchors = len(scales) * len(ratios)
    

    with tf.name_scope(name, "RPN", [x]):
        x = tf.convert_to_tensor(x, name = "input")
        # create RPN Features layer
        RPN_features_weight = tf.Variable(tf.random_normal([3, 3, in_channels, rpn_features_channels], mean=0.0, stddev = 0.01), name="features_W")
        RPN_features_bias = tf.Variable(tf.zeros([rpn_features_channels]), name = "features_bias")
        RPN_features = conv2d_batchnorm_relu(x, RPN_features_weight, RPN_features_bias, stride=1, is_training = is_training, name = "features")

        # create RPA classification layer
        RPN_cls_weight = tf.Variable(tf.random_normal([1,1,rpn_features_channels, 2 * nr_anchors], mean=0.0, stddev = 0.01), name="classification_W")
        RPN_cls_bias = tf.Variable(tf.zeros([2 * nr_anchors]), name="classification_bias")
        RPN_cls = conv2d(RPN_features, RPN_cls_weight, RPN_cls_bias, name = "classification")

        # create RPA box regression layer
        RPN_reg_weight = tf.Variable(tf.random_normal([1,1,rpn_features_channels, 4 * nr_anchors], mean=0.0, stddev = 0.01), name="regression_W")
        RPN_reg_bias = tf.Variable(tf.zeros([4 * nr_anchors]), name="regression_bias")
        RPN_reg = conv2d(RPN_features, RPN_reg_weight, RPN_reg_bias, name = "regression")
        
        confidence_list = tf.nn.softmax(tf.reshape(RPN_cls, [-1,2]))[:,1]
        reg_list = tf.reshape(RPN_reg, [-1,4])
        
        confidence_threshold = tf.placeholder_with_default(_confidence_threshold, shape = [], name = "confidenceThreshold")
        _ , bestBoxIndices = tf.nn.top_k(confidence_list, k = tf.cast(tf.maximum(tf.ones([]), tf.reduce_sum(tf.cast(tf.greater_equal(confidence_list, confidence_threshold), tf.float32))), tf.int32), sorted = True)
        best_regs = tf.gather(reg_list, bestBoxIndices)
        box_confidences = tf.gather(confidence_list, bestBoxIndices, name = "boxConfidences")
        RPN_anchors = tf.constant(rpn_generate_anchors(input_size, (x.shape[2].value,x.shape[1].value), ratios, scales), dtype = tf.float32, name = "anchors" )
        AList = tf.gather(tf.reshape(RPN_anchors, [-1,4]), bestBoxIndices)

        #predict boxes
        best_boxes = tf.identity(AList)
        best_boxes_x = tf.reshape(best_boxes[:,0] + best_regs[:,0] * AList[:,2], [-1,1])
        best_boxes_y = tf.reshape(best_boxes[:,1] + best_regs[:,1] * AList[:,3], [-1,1])
        best_boxes_w = tf.reshape(best_boxes[:,2] * tf.exp(best_regs[:,2]), [-1,1])
        best_boxes_h = tf.reshape(best_boxes[:,3] * tf.exp(best_regs[:,3]), [-1,1])
        prediction = tf.concat([best_boxes_x, best_boxes_y, best_boxes_w, best_boxes_h], 1, "boxPredictions")

    return RPN_cls, RPN_reg, confidence_threshold, box_confidences, prediction

def rpn_loss(cls, cls_groundTruth, reg, reg_groundTruth, minibatchIndices : List[int], regWeight : float = 10, name : str = None):
    '''
        Region proposal network Loss function
    
        cls - tensor of shape (-1, ?, ?, 2 * len(scales) * len(ratios)), containing the classification scores outputted by the rpn layer
        cls_groundTruth - numpy ndarray of shape (-1, 2), containing the ground truth classification scores for each of the anchors in the minibatch
        reg - tensor of shape (-1, ?, ?, 4 * len(scales) * len(ratios)), containing the regressed tuples that give the (tx, ty, tw, th) parametrized deltas of all the proposals
        reg_groundTruth - numpy ndarray of shape (-1,4), containing the ground truth parametrized deltas of the anchors in the minibatch
        minibatchIndices - a numpy array of ints containing the indices of the anchors selected for the minibatch. The inidces are in the list of anchors obtained with a reshape(-1,4) of the anchors tensor
        regWeight - the contribution factor of the regression loss to the total loss

        Returns 3 scalars:
        loss - the total rpn loss
        cls_loss - the loss for the classification of anchors (do they contain an object?)
        reg_loss - the loss for the regression of anchors (where are they placed?)

    '''
    with tf.name_scope(name, "RPN_Loss", [cls, cls_groundTruth, reg, reg_groundTruth, minibatchIndices]):
        cls = tf.convert_to_tensor(cls)
        cls_groundTruth = tf.convert_to_tensor(cls_groundTruth)
        reg = tf.convert_to_tensor(reg)
        reg_groundTruth = tf.convert_to_tensor(reg_groundTruth)
        minibatchIndices = tf.convert_to_tensor(minibatchIndices)
        # objectness loss

        # predicted objectness distribution = cls
        # ground truth objectness distribution = cls_groundTruth
        cls_minibatch = tf.gather(tf.reshape(cls, shape=[-1,2]), minibatchIndices)
        cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = cls_minibatch, labels = cls_groundTruth))

        # box regression loss
        
        # predicted anchor offsets = reg
        # ground truth anchor offsets = reg_groundTruth
        reg_minibatch = tf.gather(tf.reshape(reg, shape=[-1,4]), minibatchIndices)
        abs_deltaT = tf.abs(reg_minibatch - reg_groundTruth)
        L1smooth = tf.where(tf.less(abs_deltaT, 1.0), 0.5 * abs_deltaT * abs_deltaT, abs_deltaT - 0.5)
        reg_loss = tf.reduce_mean(cls_groundTruth[:,1] * tf.reduce_sum(L1smooth, axis=-1)) # only boxes that contain an object contribute to the loss
 
        # total loss
        loss = tf.add(cls_loss, regWeight * reg_loss, name = "loss")

        return loss, cls_loss, reg_loss

###################### neural networks ##########################
class ConvNet:

    def __init__(self, input_shape : Tuple[int, int, int], modelName : str = "neuralNetModel"):
        '''
            input_shape is in (height, width, channels) format. Use None for the height and width if you want to input any size images. Channels must be a natural number >= 1
        '''
        self.imageHeight = input_shape[0]
        self.imageWidth = input_shape[1]
        self.noOfChannels = input_shape[2]
        self.n_classes = None # should be loaded from folder
        self.prediction = None
        self.saver = None
        self.x = None
        self.y = None
        self.layer_activations = None
        self.modelName = modelName
    

    def initPlaceHolders(self):
        self.x = tf.placeholder('float',[None, self.imageHeight,self.imageWidth,self.noOfChannels],name="X")
        self.y = tf.placeholder('float',name="Y")
        self.keep_prob = tf.placeholder(tf.float32,name="keep_prob")
        self.phase = tf.placeholder(tf.bool,name="phase" )
        self.reg = tf.placeholder('float',name="reg")

    def __convolutional_neural_net(self,input_tensor, bottleneck_size, output_size, configuration, is_training, keep_prob: float):
        '''
        creates the tensorflow graph that takes as input the 'input_tensor' tensor and proceeds with the layers as described in 'configuration'
        intput_tensor is in format NHWC
        '''

        previous_layer_channels = input_tensor._shape[3].value

        layers_output = []
                
        current_layer = input_tensor

        for idx, conv_layer_def in enumerate(configuration['conv']):
            if "name" in conv_layer_def:
                layerName = conv_layer_def["name"]
            else:
                layer_name = "layer" + str(len(layers_output))
            weights_layer_name = "W_conv" + str(idx)
            biases_layer_name = "b_conv" + str(idx)

            filter_size = conv_layer_def['filter_size']            
            filter_output_channels = conv_layer_def['channels']
            filter_stride = conv_layer_def['stride']
            
            conv_weight = tf.Variable(tf.random_normal([filter_size, filter_size, previous_layer_channels, filter_output_channels]), name=weights_layer_name)
            conv_bias = tf.Variable(tf.random_normal([filter_output_channels]), name = biases_layer_name)

            if 'pooling_size' in conv_layer_def:
                ln = None
            else:
                ln = layer_name
            current_layer = conv2d_batchnorm_relu(current_layer, conv_weight, conv_bias, stride=filter_stride, is_training = is_training, name = ln)
            #max pooling
            if 'pooling_size' in conv_layer_def:
                filter_pooling_size = conv_layer_def['pooling_size']
                filter_pooling_stride = conv_layer_def['pooling_stride']
                current_layer = maxpool2d(current_layer, filter_pooling_size, filter_pooling_stride, name = layer_name)

            previous_layer_channels = filter_output_channels
            layers_output.append(current_layer)

        conv_output =  layers_output[len(configuration['conv'])-1]
        bottleneck_channels = previous_layer_channels
        bottleneck = max_pool_with_fixed_output_size(conv_output, (bottleneck_size, bottleneck_size), "bottleneck")
        layers_output.append(bottleneck)
        if output_size>0:
            conv_weight = tf.Variable(tf.random_normal([bottleneck_size, bottleneck_size, previous_layer_channels, output_size]), name="output_weights")
            conv_bias = tf.Variable(tf.random_normal([output_size]), name = "output_biases")
            classes_tensor = conv2d(bottleneck, conv_weight, conv_bias, padd_val = "VALID")
            output = tf.squeeze(classes_tensor, [1,2], name = "output")
        else:
            output = bottleneck
        return output, layers_output
         
    def build_net_up_to_bottleneck(self, configuration):
        self.initPlaceHolders()
        bottleneck, self.layer_activations = self.__convolutional_neural_net(self.x, 8, 0, configuration, self.phase, self.keep_prob)  
                
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self.session.run(init)
        return bottleneck
                                     
    def build_net(self, configuration, n_classes, weightsPath = None):          
        self.n_classes = n_classes
        self.initPlaceHolders()
        self.prediction, self.layer_activations = self.__convolutional_neural_net(self.x, 8, self.n_classes, configuration, self.phase, self.keep_prob)
        softmax = tf.nn.softmax_cross_entropy_with_logits( logits = self.prediction, labels = self.y)                                
        variables = [v for v in tf.global_variables() if v.name.startswith("W_conv") or v.name.startswith("out")  ]
        for var in variables:
            softmax = softmax + self.reg * tf.nn.l2_loss(var)

        self.cost = tf.reduce_mean(softmax,name="cost")
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost, name = "trainOptimizer")
        correct = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name = "accuracy")        
                
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self.session.run(init)
        if weightsPath is not None:            
            self.saver.restore(sess = self.session, save_path = weightsPath)                    

class RPN:
    '''
        RPN network implemented as described in https://arxiv.org/pdf/1506.01497.pdf
    '''
    
    def __init__(self):
        self.conv_feature_map_size = (8,8)
        self.anchors_aspect_ratios = [1/8.0, 1/4.0, 1/2.0, 1, 2, 4, 8]
        self.anchors_scales = [4, 8, 16, 32]
        self.nr_anchors = len(self.anchors_scales) * len(self.anchors_aspect_ratios)
        self.minibatch_size = 128
        self.ds = None # dataset in use

    def create_inputs(self):
        self.minibatch_indices = tf.placeholder(shape = [self.minibatch_size], dtype = tf.int32)
        self.gt_p = tf.placeholder(shape=[self.minibatch_size, 2], dtype = tf.float32) # ground truth objectness: see white paper
        self.gt_t = tf.placeholder(shape=[self.minibatch_size, 4], dtype = tf.float32) # ground truth parametric position: see white paper

    def create_base(self, image_shape, configuration):
        #create RPABase model
        self.current_model = ConvNet(image_shape, "RPN_Base")
        rpn_base_feature_map = self.current_model.build_net_up_to_bottleneck(configuration)

        self.conv_feature_map_size = (rpn_base_feature_map.shape[1].value, rpn_base_feature_map.shape[2].value)
        print(rpn_base_feature_map.shape)
        return rpn_base_feature_map

    def build_model(self, image_shape, base_configuration):
        #build RPABase model
        rpn_base_feature_map = self.create_base(image_shape, base_configuration)
        rpn_features_channels = 256
        self.create_inputs()
        self.rpn_cls, self.rpn_reg, self.confidence_threshold, self.box_confidences, self.box_predictions = rpn_layer(rpn_base_feature_map, (self.current_model.imageWidth, self.current_model.imageHeight), rpn_features_channels, self.anchors_scales, self.anchors_aspect_ratios, 0.5, self.current_model.phase, name = "RPN")

    def loss(self):
        self.loss, self.cls_loss, self.reg_loss = rpn_loss(self.rpn_cls, self.gt_p, self.rpn_reg, self.gt_t, self.minibatch_indices, 5, "RPN_Loss")
        return self.loss

    def train(self, ds : DetectionDataset, save_model_path : str, save_model_name : str, continue_training : bool = False):
        self.ds = ds
        L = self.loss()
        self.rpn_trainer = tf.train.AdamOptimizer().minimize(L, name = "RPN_trainer")

        self.saver = tf.train.Saver()        
        init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self.current_model.session.run(init)

        best_train_loss = -1
        best_test_loss = -1
        best_epoch = -1
        epochs = 5
        early_stop_radius = 100

        train_loss_history = []
        test_loss_history = []

        train_reg_history = []
        test_reg_history = []

        train_cls_history = []
        test_cls_history = []

        training_sets = {"train" : ds.train_set, "validation" : ds.validation_set}
        if not continue_training:
            with open(os.path.join(save_model_path,"trainingSets.json"), "w") as f:
                json.dump(training_sets, f)
        nr_epochs_to_ignore = 10
        print("Start training...", flush = True)
        for epoch in range(epochs):
            start_time = time.clock()
            print("\n\nEpoch ", epoch,":", flush = True)
            print("   Train phase", flush = True)
            # train on training data
            epoch_loss = 0
            epoch_clsLoss = 0
            epoch_regLoss = 0
            n = len(ds.train_set)
            for epoch_batch_index in range(n):
                raw_batch = ds.get_batch(1, LearningSets.trainset)
                GT = raw_batch[0][1][:, 0:-1]
                mini_batch_indices, correct_objectness, correct_anchor_offsets = rpn_get_batch(raw_batch[0][0], GT, self.conv_feature_map_size, self.anchors_aspect_ratios, self.anchors_scales, self.minibatch_size)
                epoch_x = np.reshape(raw_batch[0][0], (-1, raw_batch[0][0].shape[0], raw_batch[0][0].shape[1], raw_batch[0][0].shape[2]))
                _, cost, cls_loss, reg_loss = self.current_model.session.run([self.rpn_trainer, L, self.cls_loss, self.reg_loss], feed_dict={self.current_model.x : epoch_x, self.minibatch_indices : mini_batch_indices, self.gt_p : correct_objectness, self.gt_t : correct_anchor_offsets, self.current_model.phase: True})
                epoch_loss += cost
                epoch_clsLoss += cls_loss
                epoch_regLoss += reg_loss

            epoch_train_loss = epoch_loss / n
            print("   > Epoch train loss = ", epoch_train_loss, "  clsLoss = ", epoch_clsLoss / n, "   regLoss = ", epoch_regLoss / n, flush = True)
            if epoch >= nr_epochs_to_ignore:
                train_loss_history.append(epoch_train_loss)
                train_cls_history.append(epoch_clsLoss / n)
                train_reg_history.append(epoch_regLoss / n)

            print("   Validation phase:", flush = True)
            # test on test data
            epoch_clsLoss = 0
            epoch_regLoss = 0
            epoch_total_loss = 0
            n = len(ds.validation_set)
            for epoch_batch_index in range(n):
                raw_batch = ds.get_batch(1, LearningSets.validationset)
                GT = raw_batch[0][1][:, 0:-1]
                mini_batch_indices, correct_objectness, correct_anchor_offsets = rpn_get_batch(raw_batch[0][0], GT, self.conv_feature_map_size, self.anchors_aspect_ratios, self.anchors_scales, self.minibatch_size)
                epoch_x = np.reshape(raw_batch[0][0], (-1, raw_batch[0][0].shape[0], raw_batch[0][0].shape[1], raw_batch[0][0].shape[2]))
                cls_loss, reg_loss, total_loss = self.current_model.session.run([self.cls_loss, self.reg_loss, L], feed_dict={self.current_model.x : epoch_x, self.minibatch_indices : mini_batch_indices, self.gt_p : correct_objectness, self.gt_t : correct_anchor_offsets, self.current_model.phase: False})
                epoch_total_loss += total_loss
                epoch_clsLoss += cls_loss
                epoch_regLoss += reg_loss

            epoch_test_loss = epoch_total_loss / n
            print("   > Epoch validation loss = ", epoch_test_loss, "  clsLoss = ", epoch_clsLoss / n, "   regLoss = ", epoch_regLoss / n, flush = True)

            # save the best model we find
            if epoch >= nr_epochs_to_ignore:
                test_loss_history.append(epoch_test_loss)
                test_cls_history.append(epoch_clsLoss / n)
                test_reg_history.append(epoch_regLoss / n)
                if (best_test_loss < 0 or epoch_total_loss / n < best_test_loss):
                    best_epoch = epoch
                    best_train_loss = epoch_train_loss
                    best_test_loss = epoch_total_loss / n
                    print("    Found a better one!", end="", flush = True)
                    saved_best = True
                    print("Saving...", end="", flush = True)
                    savePath = self.saver.save(self.current_model.session, os.path.join(save_model_path, save_model_name))
                    print("Done!  Saved at:", savePath, flush = True)
            end_time = time.clock()
            print("Total epoch time = ", end_time - start_time, " seconds\n", flush = True)
            print("\n\t".join(["Best result so far:","-epoch="+str(best_epoch),"-trainLoss="+str(best_train_loss),"-testLoss="+str(best_test_loss)]))

            if (epoch - best_epoch >= early_stop_radius):
                print("*!* Early stop condition reached! earlyStopRadius = "+str(early_stop_radius))
                break
        print("\nTraining complete!\n", flush = True)

        train_history = {"trainLoss" : train_loss_history, "testLoss" : test_loss_history, "trainClsLoss" : train_cls_history, "testClsLoss" : test_cls_history, "trainRegLoss" : train_reg_history, "testRegLoss" : test_reg_history}
        with open(os.path.join(save_model_path,"trainHistory.json"), "w") as f:
            json.dump(train_history, f)


    def inference(self, img_path : str, out_directory_name : str, conf_thresold : float = 0.75):
        image = cv2.imread(img_path)
        if (image is None):
            return None
        # resize the image to fit into the network
        image = image_resize(image, (self.current_model.imageHeight, self.current_model.imageWidth))
        start_time = time.clock()
        X = np.asarray(image)
        X = np.reshape(X, (-1, X.shape[0], X.shape[1], X.shape[2]))
        conf, boxes = self.current_model.session.run([self.box_confidences, self.box_predictions], feed_dict={self.current_model.x : X, self.confidence_threshold : conf_thresold, self.current_model.phase : False})
        end_time = time.clock()
        print("Inference time = ", end_time - start_time, " seconds")
        
        image = cv2.imread(img_path)
        wf = image.shape[0] / self.current_model.imageWidth
        hf = image.shape[1] / self.current_model.imageHeight
        boxes[:,0] *= wf
        boxes[:,2] *= wf
        boxes[:,1] *= hf
        boxes[:,3] *= hf
        
        overlay = image.copy()
        for i, box in enumerate(boxes):
            cv2.rectangle(overlay, (int(np.clip(box[0], 0, image.shape[1])), int(np.clip(box[1], 0, image.shape[0]))), (int(np.clip(box[0]+box[2], 0, image.shape[1])), int(np.clip(box[1] + box[3], 0, image.shape[0]))), (int(255 * conf[i]),0,0), 1)
        opacity = 0.5
        cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)
        
        gtImage = get_training_sample_image_annotated(img_path)

        sideBySide = np.concatenate((gtImage, image), axis=1)
        dir,filename = os.path.split(img_path)
        saveDir = os.path.join(dir, out_directory_name+str(conf_thresold))
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        savePath = os.path.join(saveDir, filename)
        cv2.imwrite(savePath, sideBySide)
        #cv2.imshow(imgPath, sideBySide)
        #cv2.waitKey(0)
        return boxes

if __name__ == "__main__":
    RPN_Base =  { 'conv' : [
                    {'filter_size':3,'channels':64,'stride':1},
                    {'filter_size':3,'channels':64,'stride':1,'pooling_size':2,'pooling_stride':2}, #128
                    {'filter_size':3,'channels':64,'stride':1},                                    
                    {'filter_size':3,'channels':128,'stride':1,'pooling_size':2,'pooling_stride':2},# 64
					{'filter_size':3,'channels':128,'stride':1},
                    {'filter_size':3,'channels':128,'stride':1},
                    {'filter_size':3,'channels':256,'stride':1,'pooling_size':2,'pooling_stride':2}, #32 
                    {'filter_size':3,'channels':256,'stride':1},
                    {'filter_size':3,'channels':256,'stride':1},
                    {'filter_size':3,'channels':256,'stride':1,'pooling_size':2,'pooling_stride':2}, #16
                    ]
                }

    save_model_path = "D:\\virgil\\GoogleML_school\\models\\RPN"
    save_model_name = "RPN_model.ckpt"

    rpn = RPN()
    rpn.build_model((1024, 1024, 3), RPN_Base)
    dataset = DetectionDataset("D:\\virgil\\GoogleML_school\\dataset", 1024, 1024, 3, True)
    dataset.load()
    rpn.train(dataset, save_model_path, save_model_name)

    print("Compute demo validation", flush = True)
    for id in dataset.validation_set:
        rpn.inference(dataset.data_dict[id]['path'],"Validation", 0.995)

    print("Compute demo train", flush = True)
    for id in dataset.train_set:
        rpn.inference(dataset.data_dict[id]['path'],"Train", 0.995)

    print("Done!")
