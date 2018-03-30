import tensorflow as tf

from basic.model import Model


class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.GradientDescentOptimizer(config.init_lr)
        self.loss = model.get_loss()
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        _, ds = batch
        feed_dict = self.model.get_feed_dict(ds, True)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        opt = tf.train.GradientDescentOptimizer(learning_rate=config.init_lr)

        
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.models = models
        losses = []
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                loss = model.get_loss()
                grads = opt.compute_gradients(loss, var_list=self.var_list)
                #grads = [(tf.clip_by_value(grad, -1., 1.), var) for (grad, var) in grads if not grad is None]
                losses.append(loss)

        self.loss = tf.add_n(losses)/len(losses)
        self.train_op = opt.minimize(self.loss, global_step=self.global_step)

    def step(self, sess, batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))

        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
