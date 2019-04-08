import tensorflow as tf


categories = ["UP", "DOWN"]


def prepare(filepath):


modelpath = "models/RNN_Final-10-0.668.model"


model = tf.keras.models.load_model(modelpath)


prediction = model.predict(X,verbose=0)

print(prediction)
