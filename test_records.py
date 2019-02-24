import tensorflow as tf

def read_and_decode(filename_queue):
 reader = tf.TFRecordReader()
 _, serialized_example = reader.read(filename_queue)
 features = tf.parse_single_example(
  serialized_example,
  features={
      'image_raw': tf.FixedLenFeature([], tf.string)
  })
 image = tf.decode_raw(features['image_raw'], tf.uint8)
 return image


def get_all_records(FILE):
 with tf.Session() as sess:
   filename_queue = tf.train.string_input_producer([FILE], num_epochs=1)
   image = read_and_decode(filename_queue)
   init_op = tf.initialize_all_variables()
   sess.run(init_op)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   try:
     while True:
       example = sess.run([image])
   except tf.errors.OutOfRangeError, e:
     coord.request_stop(e)
   finally:
     coord.request_stop()
     coord.join(threads)

get_all_records('datasets/tfnabrids/tfnabirds-r02.tfrecords')
