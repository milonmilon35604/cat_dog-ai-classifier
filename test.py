import tensorflow as tf

model = tf.saved_model.load("save_model")

print("SUCCESS")