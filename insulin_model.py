import tensorflow as tf

def exponential_insulin_model(action_duration, peak_activity_time, time):
    tau = peak_activity_time * (1 - peak_activity_time / action_duration ) / (1 - 2 * peak_activity_time / action_duration)
    a = 2 * tau / action_duration
    S = 1 / (1 - a + (1 + a) * tf.exp(-action_duration / tau))
    return tf.subtract(1.0, S * (1 - a) * ((pow(time, 2) / (tau * action_duration * (1 - a)) - time / tau - 1) * tf.exp(-time / tau) + 1), name="insulin_activity")
