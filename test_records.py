import tensorflow as tf

def extract_fn(data_record):
    ex = tf.train.Example()
    ex.ParseFromString(data_record)
    shape = ex.features.feature["shape"].int64_list.value
    data = ex.features.feature["data"].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

# Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(['datasets/tfnabirds/tfnabirds-r02.tfrecords'])
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

i= 0
with tf.Session() as sess:
    while True:
        data_record = sess.run(next_element)
        print(data_record)
        img = Image.fromarray(data_record, 'RGB')
        img.save( "output/" + str(i) + '-train.png')
        i++1
