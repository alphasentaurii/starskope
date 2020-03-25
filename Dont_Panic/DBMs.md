

## AUTOENCODERS
A capsule makes much better use of the outputs of the recognition units by using them to compute precise position coordinates which are accurate to a small fraction of a pixel. I

IMG: https://en.wikipedia.org/wiki/File:Autoencoder_structure.png


### DEEP BOLTZMANN MACHINES

A transforming auto-encoder can force the outputs of a capsule to represent
any property of an image that we can manipulate in a known way. It is easy,
for example, to scale up all of the pixel intensities. If a first-level capsule outputs a number that is first multiplied by the brightness scaling factor and then
used to scale the outputs of its generation units when predicting the brightnesstransformed output, this number will learn to represent brightness and will allow
the capsule to disentangle the probability that an instance of its visual entity is
present from the brightness of the instance. If the direction of lighting of a scene
can be varied in a controlled way, a capsule can be forced to output two numbers
representing this direction but only if the visual entity is complex enough to allow the lighting direction to be extracted from the activities of the recognition
units.


Deep Boltzmann Machines (DBM’s)

http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf

We present a new learning algorithm for Boltzmann machines that contain many layers of hidden variables. Data-dependent expectations are
estimated using a variational approximation that
tends to focus on a single mode, and dataindependent expectations are approximated using persistent Markov chains. The use of two
quite different techniques for estimating the two
types of expectation that enter into the gradient
of the log-likelihood makes it practical to learn
Boltzmann machines with multiple hidden layers and millions of parameters. The learning can
be made more efficient by using a layer-by-layer
“pre-training” phase that allows variational inference to be initialized with a single bottomup pass




* a deep multilayer Boltzmann machine : each layer captures complicated, higher-order correlations between the activities of hidden features in the layer below.

* conditional distributions defined by the composed model are exactly the same conditional distributions defined by the DBM (Eqs. 11, 12, 13). Therefore greedily pretraining the two modified RBM’s leads to an undirected model with symmetric weights – a deep Boltzmann machine. 

* When greedily training a stack of more than two RBM’s, the modification only needs to be used for the first and the last RBM’s in the stack. For all the intermediate RBM’s we simply halve their weights in both directions when composing them to form a deep Boltzmann machine.



After learning, DBM is used to initialize a multilayer
neural network. The marginals of approximate posterior 
q(h
2
j =
1|v) are used as additional inputs. The network is fine-tuned by
backpropagation.




as the learning rate becomes sufficiently small
compared with the mixing rate of the Markov chain, this
“persistent” chain will always stay very close to the stationary distribution even if it is only run for a few MCMC
updates per parameter upda

3.3 Discriminative Fine-tuning of DBM’s
After learning, the stochastic activities of the binary features in each layer can be replaced by deterministic, realvalued probabilities, and a deep Boltzmann machine can be
used to initialize a deterministic multilayer neural network

For each input vector v, the meanfield inference is used to obtain an approximate posterior
distribution q(h|v). The marginals q(h
2
j = 1|v) of this
approximate posterior, together with the data, are used to
create an “augmented” input for this deep multilayer neural network as shown in Fig. 3. Standard backpropagation
can then be used to discriminatively fine-tune the model.

, the gradient-based fine-tuning may choose to ignore q(h
2
|v), i.e. drive the first-layer connections W2
to
zero, which will result in a standard neural network net.
Conversely, the network may choose to ignore the input by
driving the first-layer W1
to zero. In all of our experiments, however, the network uses the entire augmented input for making predictions.


. To speed-up learning, we subdivided datasets into
mini-batches, each containing 100 cases, and updated the
weights after each mini-batch. The number of fantasy particles used for tracking the model’s statistics was also set to
1002
. For the stochastic approximation algorithm, we always used 5 Gibbs updates of the fantasy particles. The initial learning rate was set 0.005 and was gradually decreased
to 0. For discriminative fine-tuning of DBM’s we used
the method of conjugate gradients on larger mini-batches
of 5000 with three line searches performed for each minibatch in each epoch.

The MNIST digit dataset contains 60,000 training and
10,000 test images of ten handwritten digits (0 to 9), with
28×28 pixels. In our first experiment, we trained two deep
Boltzmann machines: one with two hidden layers (500 and
1000 hidden units), and the other with three hidden layers (500, 500, and 1000 hidden units), as shown in Fig. 4.
To estimate the model’s partition function we used 20,000
βk spaced uniformly from 0 to 1.0. Table 1 shows that
the estimates of the lower bound on the average test logprobability were −84.62 and −85.18 for the 2- and 3-layer
BM’s respectively. This result is slightly better compared
to the lower bound of −85.97, achieved by a two-layer deep
belief network (Salakhutdinov and Murray, 2008).
Observe that the two DBM’s, that contain over 0.9 and
1.15 million parameters, do not appear to suffer much from
overfitting. The difference between the estimates of the
training and test log-probabilities was about 1 nat. Fig. 4
shows samples generated from the two DBM’s by randomly initializing all binary states and running the Gibbs
sampler for 100,000 steps. Certainly, all samples look
like the real handwritten digits. We also note that without
greedy pretraining, we could not successfully learn good
DBM models of MNIST digits.


------------


https://en.wikipedia.org/wiki/Autoencoder



An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner.[1] The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise”. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input, hence its name. Several variants exist to the basic model, with the aim of forcing the learned representations of the input to assume useful properties.[2] 

Examples are:
* the regularized autoencoders (Sparse, Denoising and Contractive autoencoders), proven effective in learning representations for subsequent classification tasks,[3] 
* Variational autoencoders, with their recent applications as generative models.[4] Autoencoders are effectively used for solving many applied problems, from face recognition[5] to acquiring the semantic meaning of words.[6][7]

BASIC ARCHITECTURE

The simplest form of an autoencoder is a feedforward, non-recurrent neural network similar to single layer perceptrons that participate in multilayer perceptrons (MLP) – having an input layer, an output layer and one or more hidden layers connecting them – where the output layer has the same number of nodes (neurons) as the input layer, and with the purpose of reconstructing its inputs (minimizing the difference between the input and the output) instead of predicting the target value}
 given inputs 
. Therefore, autoencoders are unsupervised learning models (do not require labeled inputs to enable learning).

An autoencoder consists of two parts, the encoder and the decoder, which can be defined as transitions}
 and}
 such that:style \phi :{\mathcal {X}}\rightarr
{\displaystyle \psi :{\mathcal {F}}\rightarrow {\mathcal {X}}}

{\displarname {arg\,min} }}\,\|X-(\psi \circ \phi )X\|^{2}}

In the simplest case, given one hidden layer, the encoder stage of an autoencoder takes the input {\displaystyle \mathbf {x} \in \mathbb {R} ^{d}={\mathcal {X}}}
 and maps it to {\displaystyle \mathbf {h} \in \mathbb {R} ^{p}={\mathcal {F}}}
:
{\displaystyle \mathbf {h} =\sigma (\mathbf {Wx} +\mathbf {b} )}

This image {\displaystyle \mathbf {h} }
 is usually referred to as code, latent variables, or latent representation. Here, {\displaystyle \sigma }
 is an element-wise activation function such as a sigmoid function or a rectified linear unit. {\displaystyle \mathbf {W} }
 is a weight matrix and {\displaystyle \mathbf {b} }
 is a bias vector. Weights and biases are usually initialized randomly, and then updated iteratively during training through Backpropagation. After that, the decoder stage of the autoencoder maps {\displaystyle \mathbf {h} }
 to the reconstruction {\displaystyle \mathbf {x'} }
 of the same shape as {\displaystyle \mathbf {x} }
:
{\displaystyle \mathbf {x'} =\sigma '(\mathbf {W'h} +\mathbf {b'} )}

where {\displaystyle \mathbf {\sigma '} ,\mathbf {W'} ,{\text{ and }}\mathbf {b'} }
 for the decoder may be unrelated to the corresponding {\displaystyle \mathbf {\sigma } ,\mathbf {W} ,{\text{ and }}\mathbf {b} }
 for the encoder.
Autoencoders are trained to minimise reconstruction errors (such as squared errors), often referred to as the "loss":
{\displaystyle {\mathcal {L}}(\mathbf {x} ,\mathbf {x'} )=\|\mathbf {x} -\mathbf {x'} \|^{2}=\|\mathbf {x} -\sigma '(\mathbf {W'} (\sigma (\mathbf {Wx} +\mathbf {b} ))+\mathbf {b'} )\|^{2}}

where {\displaystyle \mathbf {x} }
 is usually averaged over some input training set.
As mentioned before, the training of an autoencoder is performed through Backpropagation of the error, just like a regular feedforward neural network.
Should the feature space {\displaystyle {\mathcal {F}}}
 have lower dimensionality than the input space {\displaystyle {\mathcal {X}}}
, the feature vector {\displaystyle \phi (x)}
 can be regarded as a compressed representation of the input {\displaystyle x}
. This is the case of undercomplete autoencoders. If the hidden layers are larger than (overcomplete autoencoders), or equal to, the input layer, or the hidden units are given enough capacity, an autoencoder can potentially learn the identity function and become useless. However, experimental results have shown that autoencoders might still learn useful features in these cases.[13] In the ideal setting, one should be able to tailor the code dimension and the model capacity on the basis of the complexity of the data distribution to be modeled. One way to do so, is to exploit the model variants known as Regularized Autoencoders

Regularized Autoencoders[edit]

Various techniques exist to prevent autoencoders from learning the identity function and to improve their ability to capture important information and learn richer representations.
Sparse autoencoder (SAE)[edit]



Simple schema of a single-layer sparse autoencoder. The hidden nodes in bright yellow are activated, while the light yellow ones are inactive. The activation depends on the input.
Recently, it has been observed that when representations are learnt in a way that encourages sparsity, improved performance is obtained on classification tasks.[14] Sparse autoencoder may include more (rather than fewer) hidden units than inputs, but only a small number of the hidden units are allowed to be active at once.[12] This sparsity constraint forces the model to respond to the unique statistical features of the input data used for training.
Specifically, a sparse autoencoder is an autoencoder whose training criterion involves a sparsity penalty {\displaystyle \Omega ({\boldsymbol {h}})}
 on the code layer {\displaystyle {\boldsymbol {h}}}
.
{\displaystyle {\mathcal {L}}(\mathbf {x} ,\mathbf {x'} )+\Omega ({\boldsymbol {h}})}

Recalling that {\displaystyle {\boldsymbol {h}}=f({\boldsymbol {W}}{\boldsymbol {x}}+{\boldsymbol {b}})}
, the penalty encourages the model to activate (i.e. output value close to 1) some specific areas of the network on the basis of the input data, while forcing all other neurons to be inactive (i.e. to have an output value close to 0).[15]
This sparsity of activation can be achieved by formulating the penalty terms in different ways.
* One way to do it, is by exploiting the Kullback-Leibler (KL) divergence.[14][15][16][17] Let
{\displaystyle {\hat {\rho _{j}}}={\frac {1}{m}}\sum _{i=1}^{m}[h_{j}(x_{i})]}

be the average activation of the hidden unit {\displaystyle j}
 (averaged over the {\displaystyle m}
 training examples). Note that the notation {\displaystyle h_{j}(x_{i})}
 makes explicit what the input affecting the activation was, i.e. it identifies which input value the activation is function of. To encourage most of the neurons to be inactive, we would like {\displaystyle {\hat {\rho _{j}}}}
 to be as close to 0 as possible. Therefore, this method enforces the constraint {\displaystyle {\hat {\rho _{j}}}=\rho }
 where {\displaystyle \rho }
 is the sparsity parameter, a value close to zero, leading the activation of the hidden units to be mostly zero as well. The penalty term {\displaystyle \Omega ({\boldsymbol {h}})}
 will then take a form that penalizes {\displaystyle {\hat {\rho _{j}}}}
 for deviating significantly from {\displaystyle \rho }
, exploiting the KL divergence:
{\displaystyle \sum _{j=1}^{s}KL(\rho ||{\hat {\rho _{j}}})=\sum _{j=1}^{s}[\rho \log {\frac {\rho }{\hat {\rho _{j}}}}+(1-\rho )\log {\frac {1-\rho }{1-{\hat {\rho _{j}}}}}]}
 where {\displaystyle j}
 is summing over the {\displaystyle s}
 hidden nodes in the hidden layer, and {\displaystyle KL(\rho ||{\hat {\rho _{j}}})}
 is the KL-divergence between a Bernoulli random variable with mean {\displaystyle \rho }
 and a Bernoulli random variable with mean {\displaystyle {\hat {\rho _{j}}}}
.[15]
* Another way to achieve sparsity in the activation of the hidden unit, is by applying L1 or L2 regularization terms on the activation, scaled by a certain parameter {\displaystyle \lambda }.[18] For instance, in the case of L1 the loss function would become
{\displaystyle {\mathcal {L}}(\mathbf {x} ,\mathbf {x'} )+\lambda \sum _{i}|h_{i}|}

* A further proposed strategy to force sparsity in the model is that of manually zeroing all but the strongest hidden unit activations (k-sparse autoencoder).[19] The k-sparse autoencoder is based on a linear autoencoder (i.e. with linear activation function) and tied weights. The identification of the strongest activations can be achieved by sorting the activities and keeping only the first k values, or by using ReLU hidden units with thresholds that are adaptively adjusted until the k largest activities are identified. This selection acts like the previously mentioned regularization terms in that it prevents the model from reconstructing the input using too many neurons.[19]

Denoising autoencoder (DAE)[edit]

Differently from sparse autoencoders or undercomplete autoencoders that constrain representation, Denoising autoencoders (DAE) try to achieve a good representation by changing the reconstruction criterion.[2]
Indeed, DAEs take a partially corrupted input and are trained to recover the original undistorted input. In practice, the objective of denoising autoencoders is that of cleaning the corrupted input, or denoising. Two underlying assumptions are inherent to this approach:
* Higher level representations are relatively stable and robust to the corruption of the input;
* To perform denoising well, the model needs to extract features that capture useful structure in the distribution of the input.[3]
In other words, denoising is advocated as a training criterion for learning to extract useful features that will constitute better higher level representations of the input.[3]
The training process of a DAE works as follows:
* The initial input {\displaystyle x} is corrupted into {\displaystyle {\boldsymbol {\tilde {x}}}} through stochastic mapping {\displaystyle {\boldsymbol {\tilde {x}}}\thicksim q_{D}({\boldsymbol {\tilde {x}}}|{\boldsymbol {x}})}.
* The corrupted input {\displaystyle {\boldsymbol {\tilde {x}}}} is then mapped to a hidden representation with the same process of the standard autoencoder, {\displaystyle {\boldsymbol {h}}=f_{\theta }({\boldsymbol {\tilde {x}}})=s({\boldsymbol {W}}{\boldsymbol {\tilde {x}}}+{\boldsymbol {b}})}.
* From the hidden representation the model reconstructs {\displaystyle {\boldsymbol {z}}=g_{\theta '}({\boldsymbol {h}})}.[3]
The model's parameters {\displaystyle \theta }
 and {\displaystyle \theta '}
 are trained to minimize the average reconstruction error over the training data, specifically, minimizing the difference between {\displaystyle {\boldsymbol {z}}}
 and the original uncorrupted input {\displaystyle {\boldsymbol {x}}}
.[3] Note that each time a random example {\displaystyle {\boldsymbol {x}}}
 is presented to the model, a new corrupted version is generated stochastically on the basis of {\displaystyle q_{D}({\boldsymbol {\tilde {x}}}|{\boldsymbol {x}})}
.
The above-mentioned training process could be developed with any kind of corruption process. Some examples might be additive isotropic Gaussian noise, Masking noise (a fraction of the input chosen at random for each example is forced to 0) or Salt-and-pepper noise (a fraction of the input chosen at random for each example is set to its minimum or maximum value with uniform probability).[3]
Finally, notice that the corruption of the input is performed only during the training phase of the DAE. Once the model has learnt the optimal parameters, in order to extract the representations from the original data no corruption is added.
Contractive autoencoder (CAE)[edit]

Contractive autoencoder adds an explicit regularizer in their objective function that forces the model to learn a function that is robust to slight variations of input values. This regularizer corresponds to the Frobenius norm of the Jacobian matrix of the encoder activations with respect to the input. Since the penalty is applied to training examples only, this term forces the model to learn useful information about the training distribution. The final objective function has the following form:
{\displaystyle {\mathcal {L}}(\mathbf {x} ,\mathbf {x'} )+\lambda \sum _{i}||\nabla _{x}h_{i}||^{2}}

The name contractive comes from the fact that the CAE is encouraged to map a neighborhood of input points to a smaller neighborhood of output points.
There is a connection between the denoising autoencoder (DAE) and the contractive autoencoder (CAE): in the limit of small Gaussian input noise, DAE make the reconstruction function resist small but finite-sized perturbations of the input, while CAE make the extracted features resist infinitesimal perturbations of the input.




Advantages of Depth[edit]


    Schematic structure of an autoencoder with 3 fully connected hidden layers. The code (z, or h for reference in the text) is the most internal layer.


Autoencoders are often trained with only a single layer encoder and a single layer decoder, but using deep encoders and decoders offers many advantages.[2]
* Depth can exponentially reduce the computational cost of representing some functions.[2]
* Depth can exponentially decrease the amount of training data needed to learn some functions.[2]
* Experimentally, deep autoencoders yield better compression compared to shallow or linear autoencoders.[26]




    Training Deep Architectures[edit]

    Geoffrey Hinton developed a pretraining technique for training many-layered deep autoencoders. This method involves treating each neighbouring set of two layers as a restricted Boltzmann machine so that the pretraining approximates a good solution, then using a backpropagation technique to fine-tune the results.[26] This model takes the name of deep belief network.
    Recently, researchers have debated whether joint training (i.e. training the whole architecture together with a single global reconstruction objective to optimize) would be better for deep auto-encoders.[27] A study published in 2015 empirically showed that the joint training method not only learns better data models, but also learned more representative features for classification as compared to the layerwise method.[27] However, their experiments highlighted how the success of joint training for deep autoencoder architectures depends heavily on the regularization strategies adopted in the modern variants of the model.[27][28]
    Applications[edit]

    The two main applications of autoencoders since the 80s have been dimensionality reduction and information retrieval,[2] but modern variations of the basic model were proven successful when applied to different domains and tasks.
    Dimensionality Reduction[edit]



    Plot of the first two Principal Components (left) and a two-dimension hidden layer of a Linear Autoencoder (Right) applied to the Fashion MNIST dataset.[29] The two models being both linear learn to span the same subspace. The projection of the data points is indeed identical, apart from rotation of the subspace - to which PCA is invariant.








    Dimensionality Reduction was one of the first applications of deep learning, and one of the early motivations to study autoencoders.[2] In a nutshell, the objective is to find a proper projection method, that maps data from high feature space to low feature space.[2]
    One milestone paper on the subject was that of Geoffrey Hinton with his publication in Science Magazine in 2006:[26] in that study, he pretrained a multi-layer autoencoder with a stack of RBMs and then used their weights to initialize a deep autoencoder with gradually smaller hidden layers until a bottleneck of 30 neurons. The resulting 30 dimensions of the code yielded a smaller reconstruction error compared to the first 30 principal components of a PCA, and learned a representation that was qualitatively easier to interpret, clearly separating clusters in the original data.[2][26]
    Representing data in a lower-dimensional space can improve performance on different tasks, such as classification.[2] Indeed, many forms of dimensionality reduction place semantically related examples near each other,[30] aiding generalization

    Relationship with principal component analysis (PCA)[edit]





    Reconstruction of 28x28pixel images by an Autoencoder with a code size of two (two-units hidden layer) and the reconstruction from the first two Principal Components of PCA. Images come from the Fashion MNIST dataset.[29]

    If linear activations are used, or only a single sigmoid hidden layer, then the optimal solution to an autoencoder is strongly related to principal component analysis (PCA).[31][32] The weights of an autoencoder with a single hidden layer of size {\displaystyle p} (where {\displaystyle p} is less than the size of the input) span the same vector subspace as the one spanned by the first {\displaystyle p} principal components, and the output of the autoencoder is an orthogonal projection onto this subspace. The autoencoder weights are not equal to the principal components, and are generally not orthogonal, yet the principal components may be recovered from them using the singular value decomposition.[33]
    However, the potential of Autoencoders resides in their non-linearity, allowing the model to learn more powerful generalizations compared to PCA, and to reconstruct back the input with a significantly lower loss of information.[26]

    Information Retrieval benefits particularly from dimensionality reduction in that search can become extremely efficient in certain kinds of low dimensional spaces. Autoencoders were indeed applied to semantic hashing, proposed by Salakhutdinov and Hinton in 2007.[30] In a nutshell, training the algorithm to produce a low-dimensional binary code, then all database entries could be stored in a hash table mapping binary code vectors to entries. This table would then allow to perform information retrieval by returning all entries with the same binary code as the query, or slightly less similar entries by flipping some bits from the encoding of the query.



# COMPUTER VISION APPLICATIONS

The peculiar characteristics of autoencoders have rendered these model extremely useful in the processing of images for various tasks.

One example can be found in lossy image compression task, where autoencoders demonstrated their potential by outperforming other approaches and being proven competitive against JPEG 2000.[38]

Another useful application of autoencoders in the field of image preprocessing is image denoising.[39][40] The need for efficient image restoration methods has grown with the massive production of digital images and movies of all kinds, often taken in poor conditions.[41]

Autoencoders are increasingly proving their ability even in more delicate contexts such as medical imaging. In this context, they have also been used for image denoising[42] as well as super-resolution.[43] 

In the field of image-assisted diagnosis, there exist some experiments using autoencoders for the detection of breast cancer[44] or even modelling the relation between the cognitive decline of Alzheimer's Disease and the latent features of an autoencoder trained with MRI[45]

Lastly, other successful experiments have been carried out exploiting variations of the basic autoencoder for image super-resolution tasks.[46]

Machine Translation[edit]

Autoencoder has been successfully applied to the machine translation of human languages which is usually called as neural machine translation (NMT) [51][52]. In NMT, the language texts are treated as sequences to be encoded into the learning procedure, while in the decoder side the target languages will be generated. Recent years also see the application of language specific autoencoders to incorporate the linguistic features into the learning procedure, such as Chinese decomposition features 


Salakhutdinov, Ruslan; Hinton, Geoffrey (2009-07-01). "Semantic hashing". International Journal of Approximate Reasoning. Special Section on Graphical Models and Information Retrieval. 50 (7): 969–978. doi:10.1016/j.ijar.2008.11.006. ISSN 0888-613X.


1. Cho, K. (2013, February). Simple sparsification improves sparse denoising autoencoders in denoising highly corrupted images. In International Conference on Machine Learning (pp. 432-440).
2. ^ Cho, K. (2013). Boltzmann machines and denoising autoencoders for image denoising. arXiv preprint arXiv:1301.3468.
3. ^ Antoni Buades, Bartomeu Coll, Jean-Michel Morel. A review of image denoising algorithms, with a new one. Multiscale Modeling and Simulation: A SIAM Interdisciplinary Journal, Society for Industrial and Applied Mathematics, 2005, 4 (2), pp.490-530. hal-00271141
4. ^ Gondara, Lovedeep (December 2016). "Medical Image Denoising Using Convolutional Denoising Autoencoders". 2016 IEEE 16th International Conference on Data Mining Workshops (ICDMW). Barcelona, Spain: IEEE: 241–246. arXiv:1608.04667. Bibcode:2016arXiv160804667G. doi:10.1109/ICDMW.2016.0041. ISBN 9781509059102.
5. ^ Tzu-Hsi, Song; Sanchez, Victor; Hesham, EIDaly; Nasir M., Rajpoot (2017). "Hybrid deep autoencoder with Curvature Gaussian for detection of various types of cells in bone marrow trephine biopsy images". 2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017): 1040–1043. doi:10.1109/ISBI.2017.7950694. ISBN 978-1-5090-1172-8.
6. ^ Xu, Jun; Xiang, Lei; Liu, Qingshan; Gilmore, Hannah; Wu, Jianzhong; Tang, Jinghai; Madabhushi, Anant (January 2016). "Stacked Sparse Autoencoder (SSAE) for Nuclei Detection on Breast Cancer Histopathology Images". IEEE Transactions on Medical Imaging. 35 (1): 119–130. 
1.  Kyunghyun Cho; Bart van Merrienboer; Dzmitry Bahdanau; Yoshua Bengio (3 September 2014). "On the Properties of Neural Machine Translation: Encoder–Decoder Approaches". arXiv:1409.1259
2. ^ Sutskever, Ilya; Vinyals, Oriol; Le, Quoc Viet (2014). "Sequence to sequence learning with neural networks". arXiv:1409.3215