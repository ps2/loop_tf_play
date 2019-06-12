#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import insulin_model

def minutes(m, name=None):
    return tf.constant(m * 60, dtype=tf.float32, name=name)

time = tf.linspace(minutes(0), minutes(360), 72, name="time")

humalog_novolog_adult = insulin_model.exponential_insulin_model(minutes(360, "action_duration"), minutes(75, "activity_peak"), time)

print(humalog_novolog_adult)

sess = tf.Session()

activity = sess.run(humalog_novolog_adult)
print(activity)

print("********************")
print(sess.run((1-humalog_novolog_adult) * 5 * 60))

plot = plt.plot(activity)
plt.show()

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()
