import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

class RBM(tf.Module):
    def __init__(self, alpha, H, num_vis):
        self.alpha = alpha
        self.num_hid = H
        self.num_vis = num_vis
        self.errors = []
        self.energy_train = []
        self.energy_valid = ()
        self.W = tf.Variable(tf.dtypes.cast(tf.random.normal([self.num_vis, self.num_hid], stddev=0.01, dtype=tf.float32), tf.int64), name="W")
        self.vb = tf.Variable(tf.zeros([self.num_vis]), name="vb")
        self.hb = tf.Variable(tf.zeros([self.num_hid]), name="hb")

    def gibbs_sampling(self, v):
        v = tf.dtypes.cast(v, tf.int64)
        h = tf.nn.sigmoid(tf.matmul(v, self.W) + self.hb)
        h = tf.nn.relu(tf.sign(h - tf.random.uniform(tf.shape(h))))

        v_ = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vb)
        v_ = tf.nn.relu(tf.sign(v_ - tf.random.uniform(tf.shape(v_))))

        return h, v_

    def free_energy(self, v):
        vb_term = tf.matmul(v, tf.reshape(self.vb, [-1, 1]))
        hidden_term = tf.reduce_sum(tf.math.log(1 + tf.exp(tf.matmul(v, self.W) + self.hb)), axis=1)
        return -hidden_term - vb_term

    def update_weights(self, v0, vk, h0, hk):
        w_pos_grad = tf.matmul(tf.transpose(v0), h0)
        w_neg_grad = tf.matmul(tf.transpose(vk), hk)
        CD = (w_pos_grad - w_neg_grad) / tf.cast(tf.shape(v0)[0], tf.float32)
        return self.W.assign_add(self.alpha * CD), \
               self.vb.assign_add(self.alpha * tf.reduce_mean(v0 - vk, 0)), \
               self.hb.assign_add(self.alpha * tf.reduce_mean(h0 - hk, 0))

    def training_step(self, v):
        h0, vk = self.gibbs_sampling(v)
        hk, _ = self.gibbs_sampling(vk)
        update_op = self.update_weights(v, vk, h0, hk)
        err = v - vk
        err_sum = tf.reduce_mean(err * err)
        return update_op, err_sum

    def training(self, train, valid, user, epochs, batchsize, free_energy, verbose, filename):
        print("Shape of Training Data:", train.shape)

        print("Training RBM with {0} epochs and batch size: {1}".format(epochs, batchsize))
        print("Starting the training process")

        for i in range(epochs):
            for start, end in zip(range(0, len(train), batchsize), range(batchsize, len(train), batchsize)):
                batch = train[start:end]
                update_op, err_sum = self.training_step(batch)
                cur_err = err_sum.numpy()

            self.errors.append(cur_err)

            if valid:
                etrain = np.mean(self.free_energy(train))
                self.energy_train.append(etrain)
                evalid = np.mean(self.free_energy(valid))
                self.energy_valid.append(evalid)

            if verbose:
                print("Error after {0} epochs is: {1}".format(i+1, cur_err))
            elif i % 10 == 9:
                print("Error after {0} epochs is: {1}".format(i+1, cur_err))

        if not os.path.exists('rbm_models'):
            os.mkdir('rbm_models')
        filename = 'rbm_models/'+filename
        if not os.path.exists(filename):
            os.mkdir(filename)

        np.save(filename+'/w.npy', self.W.numpy())
        np.save(filename+'/vb.npy', self.vb.numpy())
        np.save(filename+'/hb.npy', self.hb.numpy())

        if free_energy:
            print("Exporting free energy plot")
            self.export_free_energy_plot(filename)
        print("Exporting errors vs epochs plot")
        self.export_errors_plot(filename)

        inputUser = [train[user]]
        hh0, vv1 = self.gibbs_sampling(inputUser)
        feed = hh0.numpy()
        rec = vv1.numpy()
        return rec, self.W.numpy(), self.vb.numpy(), self.hb.numpy()

    def load_predict(self, filename, train, user):
        prv_w = np.load('rbm_models/'+filename+'/w.npy')
        prv_vb = np.load('rbm_models/'+filename+'/vb.npy')
        prv_hb = np.load('rbm_models/'+filename+'/hb.npy')

        inputUser = [train[user]]

        hh0, vv1 = self.gibbs_sampling(inputUser)
        feed = hh0.numpy()
        rec = vv1.numpy()

        return rec, prv_w, prv_vb, prv_hb

    def calculate_scores(self, ratings, attractions, rec, user):
        '''
        Function to obtain recommendation scores for a user
        using the trained weights
        '''
        # Creating recommendation score for books in our data
        ratings["Recommendation Score"] = rec[0]

        """ Recommend User what books he has not read yet """
        # Find the mock user's user_id from the data
        cur_user_id = ratings[ratings['user_id'] == user]['user_id'].tolist()[0]

        # Find all books the mock user has read before
        read_books = ratings[ratings['user_id'] == cur_user_id]['attraction_id']
        read_books

        # converting the pandas series object into a list
        read_books_list = read_books.tolist()

        # getting the book names and authors for the books already read by the user
        read_books_names = []
        for book in read_books_list:
            read_books_names.append(
                attractions[attractions['attraction_id'] == book]['name'].tolist()[0])

        # Find all books the mock user has 'not' read before using the to_read data
        unread_books = attractions[~attractions['attraction_id'].isin(read_books_list)]['attraction_id']
        unread_books_list = unread_books.tolist()

        # extract the ratings of all the unread books from ratings dataframe
        unread_with_score = ratings[ratings['attraction_id'].isin(unread_books_list)]

        # grouping the unread data on book id and taking the mean of the recommendation scores for each book_id
        grouped_unread = unread_with_score.groupby('attraction_id', as_index=False)['Recommendation Score'].mean()

        # getting the names and authors of the unread books
        unread_books_names = []
        for book in grouped_unread['attraction_id'].tolist():
            unread_books_names.append(
                attractions[attractions['attraction_id'] == book]['name'].tolist()[0])

        # creating a data frame for unread books with their names and recommendation scores
        unread_books_df = pd.DataFrame({
            'attraction_id': grouped_unread['attraction_id'].tolist(),
            'name': unread_books_names,
            'score': grouped_unread['Recommendation Score'].tolist()
        })

        # creating a data frame for read books with the names
        read_books_df = pd.DataFrame({
            'attraction_id': read_books_list,
            'name': read_books_names
        })

        return unread_books_df, read_books_df

    def export(self, unread_books_df, read_books_df, filename, user):
        '''
        Function to export the final result for a user into csv format
        '''
        # sort the result in descending order of the recommendation score
        sorted_result = unread_books_df.sort_values(by='score', ascending=False)

        x = sorted_result[['score']].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler((0, 5))
        x_scaled = min_max_scaler.fit_transform(x)

        sorted_result['score'] = x_scaled

        # exporting the read and unread books with scores to csv files
        read_books_df.to_csv(filename+'/user'+str(user)+'_read.csv', index=False)
        sorted_result.to_csv(filename+'/user'+str(user)+'_unread.csv', index=False)

    def export_errors_plot(self, filename):
        plt.plot(self.errors)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.savefig(filename+"/error.png")

    def export_free_energy_plot(self, filename):
        fig, ax = plt.subplots()
        ax.plot(self.energy_train, label='train')
        ax.plot(self.energy_valid, label='valid')
        leg = ax.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Free Energy")
        plt.savefig(filename+"/free_energy.png")
