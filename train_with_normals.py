import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import h5py
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_with_normals_improved_3_classes', help='Model name [default: pointnet_with_normals_improved_3_classes]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--momentum', type=float, default=0.95, help='Initial momentum [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='momentum or adam [default: adam]')
parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate for lr decay [default: 0.95]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 300]')
parser.add_argument('--decay_step', type=int, default=20000, help='Decay step for lr decay [default: 20000]')
parser.add_argument('--min_class_acc', type=float, default=0.65, help='Minimum accuracy per class to save model [default: 0.65]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MIN_CLASS_ACC = FLAGS.min_class_acc

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

# Create log directory
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.model + '_enhanced_saving_' + timestr)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_with_normals.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def augment_batch_data(batch_data, batch_normals):
    rotated_data = provider.rotate_point_cloud(batch_data)
    rotated_normals = provider.rotate_point_cloud(batch_normals)
    jittered_data = provider.jitter_point_cloud(rotated_data)
    
    # Scale points
    scale = np.random.uniform(0.8, 1.2, (BATCH_SIZE, 1, 1))
    scaled_data = jittered_data * scale
    
    # No need to scale normals as they represent directions
    return scaled_data, rotated_normals

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, normals_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, normals_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.compat.v1.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.cast(labels_pl, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver()
            # Dedicated saver for best checkpoints that should never be deleted/rotated
            best_saver = tf.compat.v1.train.Saver(max_to_keep=1000000, name='best_saver')
            # Saver for high-quality models (all classes > 0.7 accuracy)
            quality_saver = tf.compat.v1.train.Saver(max_to_keep=1000000, name='quality_saver')
        
        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)

        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'normals_pl': normals_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        best_epoch = -1
        quality_models_saved = 0  # Counter for quality models saved
        
        # Load data files
        #TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'Pre_Processed_data/3_classes/Validation/Augmented/HDF5/train_files.txt'))
        #TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'Pre_Processed_data/3_classes/Validation/Augmented/HDF5/test_files.txt'))
        TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'Pre_Processed_data/3_classes/Validation/Mixed/HDF5/train_files.txt'))
        TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'Pre_Processed_data/3_classes/Validation/Mixed/HDF5/test_files.txt'))
        #TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'Pre_Processed_data/3_classes/Simulated/HDF5/train_files.txt'))
        #TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'Pre_Processed_data/3_classes/Simulated/HDF5/test_files.txt'))
        
        # Load all training data
        train_points = []
        train_normals = []
        train_labels = []
        for fn in range(len(TRAIN_FILES)):
            print('Loading %s' % TRAIN_FILES[fn])
            with h5py.File(TRAIN_FILES[fn], 'r') as f:
                points = f['data'][:]
                normals = f['normal'][:]
                labels = f['label'][:]
                train_points.append(points)
                train_normals.append(normals)
                train_labels.append(labels)
        train_points = np.concatenate(train_points, axis=0)
        train_normals = np.concatenate(train_normals, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        
        # Convert per-point labels to per-cloud labels if needed
        if len(train_labels.shape) > 1 and train_labels.shape[1] == NUM_POINT:
            cloud_labels = []
            for i in range(train_labels.shape[0]):
                unique, counts = np.unique(train_labels[i], return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                cloud_labels.append(most_common_label)
            train_labels = np.array(cloud_labels)
        
        # Load all test data
        test_points = []
        test_normals = []
        test_labels = []
        for fn in range(len(TEST_FILES)):
            print('Loading %s' % TEST_FILES[fn])
            with h5py.File(TEST_FILES[fn], 'r') as f:
                points = f['data'][:]
                normals = f['normal'][:]
                labels = f['label'][:]
                test_points.append(points)
                test_normals.append(normals)
                test_labels.append(labels)
        test_points = np.concatenate(test_points, axis=0)
        test_normals = np.concatenate(test_normals, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        
        # Convert per-point labels to per-cloud labels if needed
        if len(test_labels.shape) > 1 and test_labels.shape[1] == NUM_POINT:
            cloud_labels = []
            for i in range(test_labels.shape[0]):
                unique, counts = np.unique(test_labels[i], return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                cloud_labels.append(most_common_label)
            test_labels = np.array(cloud_labels)
            
        print('train points shape:', train_points.shape)
        print('train normals shape:', train_normals.shape)
        print('train labels shape:', train_labels.shape)
        print('test points shape:', test_points.shape)
        print('test normals shape:', test_normals.shape)
        print('test labels shape:', test_labels.shape)

        # Print class distribution
        print('\n=== Class Distribution ===')
        unique, counts = np.unique(train_labels, return_counts=True)
        for i, (class_id, count) in enumerate(zip(unique, counts)):
            print(f'Class {class_id}: {count} samples ({count/len(train_labels)*100:.1f}%)')
        
        unique, counts = np.unique(test_labels, return_counts=True)
        print('\nTest set:')
        for i, (class_id, count) in enumerate(zip(unique, counts)):
            print(f'Class {class_id}: {count} samples ({count/len(test_labels)*100:.1f}%)')

        log_string(f'=== ENHANCED SAVING CONFIGURATION ===')
        log_string(f'Minimum class accuracy threshold: {MIN_CLASS_ACC}')
        log_string(f'Will save ALL models where ALL 3 classes have accuracy > {MIN_CLASS_ACC}')
        log_string(f'=====================================')

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_points, train_normals, train_labels, train_writer)
            eval_acc, class_accuracies = eval_one_epoch(sess, ops, test_points, test_normals, test_labels, test_writer)
            
            # Check if all classes meet the minimum accuracy threshold
            all_classes_good = all(acc >= MIN_CLASS_ACC for acc in class_accuracies)
            
            # Save the best overall model
            if eval_acc > best_acc:
                best_acc = eval_acc
                best_epoch = epoch
                # Keep latest best as a convenience symlink-like name (no rotation)
                save_path = best_saver.save(sess, os.path.join(LOG_DIR, "best_model.ckpt"), write_state=False)
                log_string("Best model saved in file: %s" % save_path)
                # Also save a unique, non-overwriting checkpoint for this best
                unique_prefix = os.path.join(LOG_DIR, "best_model_epoch_%03d_acc_%.4f.ckpt" % (epoch, eval_acc))
                unique_save_path = best_saver.save(sess, unique_prefix, write_state=False)
                log_string("Unique best model saved in file: %s" % unique_save_path)
                # Write/update best model info
                info_path = os.path.join(LOG_DIR, "best_model_info.txt")
                try:
                    with open(info_path, "w") as f_info:
                        f_info.write("epoch: %d\n" % best_epoch)
                        f_info.write("eval_accuracy: %.6f\n" % best_acc)
                        f_info.write("class_accuracies: %s\n" % str(class_accuracies))
                        f_info.write("latest_best_path: %s\n" % save_path)
                        f_info.write("unique_best_path: %s\n" % unique_save_path)
                        f_info.write("saved_at: %s\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                except Exception as e:
                    log_string("Failed to write best_model_info.txt: %s" % str(e))
            
            # Save quality models (all classes > MIN_CLASS_ACC)
            if all_classes_good:
                try:
                    quality_models_saved += 1
                    quality_prefix = os.path.join(LOG_DIR, "quality_epoch_%03d_acc_%.3f_c%.2f%.2f%.2f.ckpt" % 
                                                (epoch, eval_acc, class_accuracies[0], class_accuracies[1], class_accuracies[2]))
                    quality_save_path = quality_saver.save(sess, quality_prefix, write_state=False)
                    log_string("QUALITY MODEL SAVED: Epoch %d - Overall: %.4f, Classes: [%.3f, %.3f, %.3f]" % 
                              (epoch, eval_acc, class_accuracies[0], class_accuracies[1], class_accuracies[2]))
                    log_string("Quality model saved in file: %s" % quality_save_path)
                    
                    # Update quality models info
                    quality_info_path = os.path.join(LOG_DIR, "quality_models_info.txt")
                    try:
                        with open(quality_info_path, "a") as f_info:
                            f_info.write("epoch: %d, overall_acc: %.6f, class_accs: [%.6f, %.6f, %.6f], path: %s\n" % 
                                       (epoch, eval_acc, class_accuracies[0], class_accuracies[1], class_accuracies[2], quality_save_path))
                    except Exception as e:
                        log_string("Failed to write quality_models_info.txt: %s" % str(e))
                except Exception as e:
                    log_string("ERROR: Failed to save quality model at epoch %d: %s" % (epoch, str(e)))
                    log_string("Training will continue despite quality model save failure")
            else:
                log_string("SKIP Epoch %d - Classes: [%.3f, %.3f, %.3f] - Not all classes meet %.2f threshold" % 
                          (epoch, class_accuracies[0], class_accuracies[1], class_accuracies[2], MIN_CLASS_ACC))
            
            # Save the model periodically (every 20 epochs)
            if epoch % 20 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
                log_string("Periodic model saved in file: %s" % save_path)
        
        # Save the last epoch model
        last_epoch_save_path = saver.save(sess, os.path.join(LOG_DIR, "last_epoch_model.ckpt"), global_step=epoch)
        log_string("Last epoch model saved in file: %s" % last_epoch_save_path)
        
        # Final summary
        log_string(f'=== TRAINING COMPLETED ===')
        log_string(f'Total quality models saved: {quality_models_saved}')
        log_string(f'Best overall accuracy: {best_acc:.4f} at epoch {best_epoch}')
        log_string(f'Quality models saved in: {LOG_DIR}')
        log_string(f'========================')

def train_one_epoch(sess, ops, train_points, train_normals, train_labels, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # BALANCED SAMPLING: Ensure each class is represented equally in each batch
    num_batches = len(train_points) // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    # Group samples by class for balanced sampling
    class_indices = {0: [], 1: [], 2: []}
    for i, label in enumerate(train_labels):
        class_indices[label].append(i)
    
    # Convert to numpy arrays for easier sampling
    for class_id in class_indices:
        class_indices[class_id] = np.array(class_indices[class_id])
    
    # Calculate samples per class per batch (balanced)
    samples_per_class = BATCH_SIZE // 3  # 10-11 samples per class in a batch of 32
    
    for batch_idx in range(num_batches):
        batch_indices = []
        
        # Sample equally from each class
        for class_id in range(3):
            if len(class_indices[class_id]) > 0:
                # Sample with replacement if we don't have enough samples
                if len(class_indices[class_id]) >= samples_per_class:
                    selected = np.random.choice(class_indices[class_id], samples_per_class, replace=False)
                else:
                    selected = np.random.choice(class_indices[class_id], samples_per_class, replace=True)
                batch_indices.extend(selected)
        
        # If we need more samples, fill with random samples
        while len(batch_indices) < BATCH_SIZE:
            random_idx = np.random.randint(0, len(train_points))
            batch_indices.append(random_idx)
        
        # Shuffle the batch indices
        np.random.shuffle(batch_indices)
        batch_indices = batch_indices[:BATCH_SIZE]  # Ensure exact batch size
        
        batch_data = train_points[batch_indices]
        batch_normals = train_normals[batch_indices]
        batch_label = train_labels[batch_indices]
        
        # Augment batched point clouds by rotation, jittering, and scaling
        aug_data, aug_normals = augment_batch_data(batch_data, batch_normals)
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['normals_pl']: aug_normals,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:BATCH_SIZE] == batch_label[0:BATCH_SIZE])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        
        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0

def eval_one_epoch(sess, ops, test_points, test_normals, test_labels, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(3)]
    total_correct_class = [0 for _ in range(3)]
    
    num_batches = len(test_points) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        
        batch_data = test_points[start_idx:end_idx]
        batch_normals = test_normals[start_idx:end_idx]
        batch_label = test_labels[start_idx:end_idx]
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['normals_pl']: batch_normals,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:BATCH_SIZE] == batch_label[0:BATCH_SIZE])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        
        for i in range(BATCH_SIZE):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    
    # Debug class distribution
    log_string('Class distribution: Class 0: %d, Class 1: %d, Class 2: %d' % 
               (total_seen_class[0], total_seen_class[1], total_seen_class[2]))
    log_string('Class correct: Class 0: %d, Class 1: %d, Class 2: %d' % 
               (total_correct_class[0], total_correct_class[1], total_correct_class[2]))
    
    # Calculate class accuracy, handling missing classes (avoid division by zero)
    class_accuracies = []
    for i in range(3):
        if total_seen_class[i] > 0:
            class_acc = total_correct_class[i] / float(total_seen_class[i])
            class_accuracies.append(class_acc)
            log_string('Class %d accuracy: %f (%d/%d)' % (i, class_acc, total_correct_class[i], total_seen_class[i]))
        else:
            class_accuracies.append(0.0)  # No samples for this class
            log_string('Class %d accuracy: N/A (no samples)' % i)
    
    log_string('eval avg class acc: %f' % (np.mean(class_accuracies)))
    EPOCH_CNT += 1
    return total_correct/float(total_seen), class_accuracies

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train() 
