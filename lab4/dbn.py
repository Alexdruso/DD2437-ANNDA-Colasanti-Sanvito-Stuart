from rbm import RestrictedBoltzmannMachine
from util import *


class DeepBeliefNet:
    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''

    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {

            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                   is_bottom=True, image_size=image_size, batch_size=batch_size),

            'hid--pen': RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"],
                                                   batch_size=batch_size),

            'pen+lbl--top': RestrictedBoltzmannMachine(ndim_visible=sizes["pen"] + sizes["lbl"],
                                                       ndim_hidden=sizes["top"],
                                                       is_top=True, n_labels=n_labels, batch_size=batch_size)
        }

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size

        self.n_gibbs_recog = 15

        self.n_gibbs_gener = 200

        self.n_gibbs_wakesleep = 5

        self.print_period = 2000

    def recognize(self, true_images, true_labels):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_images: visible data shaped (number of samples, size of visible layer)
          true_labels: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        vis = true_images  # visible layer gets the image data

        labels = np.ones(true_labels.shape) / 10.  # start the net by telling you know nothing about labels

        # finished
        # fix the image data in the visible layer and drive the network bottom to top. In the top
        #  RBM, run alternating Gibbs sampling \ and read out the labels (replace pass below and 'predicted_lbl' to
        #  your predicted labels). NOTE : inferring entire train/test set may require too much compute memory (
        #  depends on your system). In that case, divide into mini-batches.

        h_1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[1]
        h_2 = self.rbm_stack['hid--pen'].get_h_given_v_dir(h_1)[1]
        h_2_label = np.concatenate((h_2, labels), axis=1)
        for _ in range(self.n_gibbs_recog):
            out = self.rbm_stack["pen+lbl--top"].get_h_given_v(h_2_label)[1]
            h_2_label = self.rbm_stack["pen+lbl--top"].get_v_given_h(out)[1]

            # fix the image
            h_2_label[:, :-labels.shape[1]:] = h_2

        predicted_lbl = h_2_label[:, -true_labels.shape[1]:]
        predicted_list = []
        for p_l in predicted_lbl:
            predicted_list.append(np.where(p_l == 1)[0])

        print("accuracy = %.2f%%" % (100. * np.mean(np.argmax(predicted_lbl, axis=1) == np.argmax(true_labels, axis=1))))

    def generate(self, true_label, name):

        """Generate data from labels

        Args:
          true_label: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        records = []
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        label = true_label

        # fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From
        #  the top RBM, drive the network \ top to the bottom visible layer (replace 'vis' from random to your
        #  generated visible layer).

        random_vis = np.random.choice([0, 1], self.sizes['vis']).reshape(-1, self.sizes['vis'])
        h_1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(random_vis)[1]
        h_2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(h_1)[1]
        h_2_label = np.concatenate((h_2, label), axis=1)

        # from top to bottom
        for _ in range(self.n_gibbs_gener):
            top = self.rbm_stack["pen+lbl--top"].get_h_given_v(h_2_label)[1]
            h_2_label = self.rbm_stack["pen+lbl--top"].get_v_given_h(top)[1]
            # fix the labels
            h_2_label[:, -label.shape[1]:] = label[:, :]
            h_2_top_to_bot = h_2_label[:, :-label.shape[1]]
            h_1_top_to_bot = self.rbm_stack["hid--pen"].get_v_given_h_dir(h_2_top_to_bot)[1]

            vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_1_top_to_bot)[1]

            records.append([ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True,
                                      interpolation=None)])

        stitch_video(fig, records).save("%s.generate%d.mp4" % (name, np.argmax(true_label)))
        # plot_images(np.array(records), np.arange(0, 10)[int((np.where(true_lbl == 1))[0])] * np.ones(len((records))))
        return records


def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try:

            self.loadfromfile_rbm(loc="trained_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()

            self.loadfromfile_rbm(loc="trained_rbm", name="pen+lbl--top")

        except IOError:

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily

            print("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")

            print("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """
            self.rbm_stack["vis--hid"].untwine_weights()
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")

            print("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            self.rbm_stack["hid--pen"].untwine_weights()
            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")


    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")

        try:

            self.loadfromfile_dbn(loc="trained_dbn", name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn", name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn", name="pen+lbl--top")

        except IOError:

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):

                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0: print("iteration=%7d" % it)

            self.savetofile_dbn(loc="trained_dbn", name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn", name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn", name="pen+lbl--top")

        return

    def loadfromfile_rbm(self, loc, name):

        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_rbm(self, loc, name):

        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self, loc, name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy" % (loc, name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_dbn(self, loc, name):

        np.save("%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return
