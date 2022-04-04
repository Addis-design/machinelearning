import tensorflow as tf
cluster_spec = tf.train.ClusterSpec({'worker' : ['localhost:2222']})
server = tf.train.Server(cluster_spec)
server.target
server.server_def
sess = tf.Session(target=server.target)
server = tf.train.Server.create_local_server()
devices = sess.list_devices()
for d in devices:
    print(d.name)
sess.close()