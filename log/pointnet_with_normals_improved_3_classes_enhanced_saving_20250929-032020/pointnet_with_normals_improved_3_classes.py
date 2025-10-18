import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    normals_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size,))
    return pointclouds_pl, normals_pl, labels_pl

def get_model(point_cloud, normals, is_training, bn_decay=None):
    """ Classification PointNet with normals, input is BxNx3 points and BxNx3 normals, output Bx3 """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    end_points = {}
    
    # Normalize point cloud to unit sphere
    point_cloud_normalized = tf.divide(
        point_cloud - tf.reduce_mean(point_cloud, axis=1, keepdims=True),
        tf.maximum(
            tf.sqrt(tf.reduce_sum(tf.square(point_cloud), axis=2, keepdims=True)),
            1e-8
        )
    )
    
    # Normalize normal vectors (ensure unit length)
    normals_normalized = tf.nn.l2_normalize(normals, axis=2)
    
    # Concatenate points and normals
    point_normal_cloud = tf.concat([point_cloud_normalized, normals_normalized], axis=2)
    
    print(f"Input combined shape: {point_normal_cloud.shape}")
    
    # Input transformation (6D: 3D points + 3D normals)
    with tf.compat.v1.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_normal_cloud, is_training, bn_decay, K=6)
    point_normal_transformed = tf.matmul(point_normal_cloud, transform)
    
    # Point functions (MLP implemented as conv2d)
    net = tf.expand_dims(point_normal_transformed, -1)
    
    # First set of layers
    net = tf_util.conv2d(net, 64, [1,6], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    
    # Feature transform
    with tf.compat.v1.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])
    
    # Second set of layers
    net = tf_util.conv2d(point_feat, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    
    # Global features
    global_feat = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool')
    
    # Fully connected layers with dropout
    net = tf.reshape(global_feat, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
    
    # Final layer (3 classes)
    with tf.compat.v1.variable_scope('fc3') as sc:
        weights = tf.compat.v1.get_variable('weights', [256, 3],
                                          initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01))
        biases = tf.compat.v1.get_variable('biases', [3], 
                                         initializer=tf.compat.v1.constant_initializer(0.0))
        net = tf.matmul(net, weights) + biases
    
    print(f"Output prediction shape: {net.shape}")
    
    return net, end_points

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ Improved loss function with better focal loss parameters """
    # Add epsilon to avoid numerical instability
    epsilon = 1e-10
    pred = tf.clip_by_value(pred, -1e3, 1e3)
    
    # Focal loss for better handling of class imbalance
    # Conservative parameters for better convergence
    alpha = 0.5  # Balanced focus on all classes
    gamma = 2.0  # Moderate focus on hard examples
    
    # Calculate focal loss
    probs = tf.nn.softmax(pred + epsilon)
    one_hot_labels = tf.one_hot(label, depth=3)
    
    # Focal loss calculation
    pt = tf.reduce_sum(probs * one_hot_labels, axis=1)
    focal_weight = tf.pow(1 - pt, gamma)
    focal_loss = -alpha * focal_weight * tf.reduce_sum(one_hot_labels * tf.math.log(probs + epsilon), axis=1)
    focal_loss = tf.reduce_mean(focal_loss)
    
    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform']
    K = transform.get_shape()[1]
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    
    return focal_loss + reg_weight * mat_diff_loss

def input_transform_net(point_cloud, is_training, bn_decay=None, K=6):
    """ Input (XYZ + Normals) Transform Net, input is BxNx6
        Return:
            Transformation matrix of size 6x6 """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,K],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_XYZ') as sc:
        weights = tf.compat.v1.get_variable('weights', [256, K*K],
                                          initializer=tf.zeros_initializer(),
                                          dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K*K],
                                         initializer=tf.zeros_initializer(),
                                         dtype=tf.float32)
        identity = tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)
        transform = transform + identity

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform

def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0]
    num_point = inputs.get_shape()[1]

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_feat') as sc:
        weights = tf.compat.v1.get_variable('weights', [256, K*K],
                                          initializer=tf.zeros_initializer(),
                                          dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K*K],
                                         initializer=tf.zeros_initializer(),
                                         dtype=tf.float32)
        identity = tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)
        transform = transform + identity

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
