# Training Deep Architectures

RECONSTRUCTION
* DIMENSIONALITY REDUCTION
* INFORMATION RETRIEVAL

    * reduce the computational cost of representing some functions.

    * Single layer of depth (encoder, decoder)

    * exponentially decrease the amount of training data needed to learn some functions.
    
    * yield better compression compared to shallow or linear autoencoders

IMAGES:
* SUPER-RESOLUTION
* DE-NOISING




# Applications

## Dimensionality Reduction

Plot of the first two Principal Components (left) and a two-dimension hidden layer of a Linear Autoencoder (Right) applied to the Fashion MNIST dataset.[29] The two models being both linear learn to span the same subspace. The projection of the data points is indeed identical, apart from rotation of the subspace - to which PCA is invariant.

Dimensionality Reduction
*In a nutshell, the objective is to find a proper projection method, that maps data from high feature space to low feature space.*

Geoffrey Hinton : publication in Science Magazine in 2006:[26]  
pretrained a multi-layer autoencoder with a stack of RBMs and then used their weights to initialize a deep autoencoder with gradually smaller hidden layers until a bottleneck of 30 neurons. 
The resulting 30 dimensions of the code yielded a smaller reconstruction error compared to the first 30 principal components of a PCA, and learned a representation that was qualitatively easier to interpret, clearly separating clusters in the original data.[2][26]

Representing data in a lower-dimensional space can improve performance on different tasks, such as classification.[2] Indeed, many forms of dimensionality reduction place semantically related examples near each other,[30] aiding generalization

Relationship with principal component analysis (PCA)[edit]


## information retrieval

Reconstruction of 28x28pixel images by an Autoencoder with a code size of two (two-units hidden layer) and the reconstruction from the first two Principal Components of PCA. Images come from the Fashion MNIST dataset.[29]

If linear activations are used, or only a single sigmoid hidden layer, then the optimal solution to an autoencoder is strongly related to principal component analysis (PCA). 

### WEIGHTS: single hidden layer, same vector subspace
The weights of an autoencoder with a single hidden layer of size {\displaystyle p} (where {\displaystyle p} is less than the size of the input) span the same vector subspace as the one spanned by the first {\displaystyle p} principal components, and the output of the autoencoder is an orthogonal projection onto this subspace. 

The autoencoder weights are not equal to the principal components, and are generally not orthogonal, *yet the principal components may be recovered from them using the `singular value decomposition`.*

### More powerful than PCA: *Non-Linearity*
* more powerful generalizations compared to PCA
* significantly lower loss of information (reconstruction)

However, the potential of Autoencoders resides in their non-linearity, allowing the model to learn more powerful generalizations compared to PCA, and to reconstruct back the input with a significantly lower loss of information.[26]

Information Retrieval benefits particularly from dimensionality reduction in that search can become extremely efficient in certain kinds of low dimensional spaces. Autoencoders were indeed applied to semantic hashing, proposed by Salakhutdinov and Hinton in 2007.[30] In a nutshell, training the algorithm to produce a low-dimensional binary code, then all database entries could be stored in a hash table mapping binary code vectors to entries. This table would then allow to perform information retrieval by returning all entries with the same binary code as the query, or slightly less similar entries by flipping some bits from the encoding of the query.


**variations of the basic model for different domains and tasks**


# Deep Belief Network
## pretraining technique for training many-layered deep autoencoders. 

This method involves treating each neighbouring set of two layers as a restricted Boltzmann machine so that the pretraining approximates a good solution, then using a backpropagation technique to fine-tune the results. This model takes the name of deep belief network.

# joint training 
    training the whole architecture together with a single global reconstruction objective to optimize. 
    
    the joint training method not only learns better data models, but also learned more representative features for classification as compared to the layerwise method.
    
    However, the success of joint training for deep autoencoder architectures *depends heavily on the regularization strategies adopted in the modern variants of the model.* 

Advantages of Depth


    Schematic structure of an autoencoder with 3 fully connected hidden layers. The code (z, or h for reference in the text) is the most internal layer.


Autoencoders are often trained with only a single layer encoder and a single layer decoder, but using deep encoders and decoders offers many advantages.[2]
* Depth can exponentially reduce the computational cost of representing some functions.[2]
* Depth can exponentially decrease the amount of training data needed to learn some functions.[2]
* Experimentally, deep autoencoders yield better compression compared to shallow or linear autoencoders.[26]