# ON THE RELATIONSHIP BETWEEN SELF-ATTENTION AND CONVOLUTIONAL LAYERS  

Jean-Baptiste Cordonnier, Andreas Loukas & Martin Jaggi Ecole Polytechnique Fédérale de Lausanne (EPFL) {first.last}@epfl.ch  

# ABSTRACT  

Recent trends of incorporating attention mechanisms in vision have led researchers to reconsider the supremacy of convolutional layers as a primary building block. Beyond helping CNNs to handle long-range dependencies, Ramachandran et al. (2019) showed that attention can completely replace convolution and achieve state-of-the-art performance on vision tasks. This raises the question: do learned attention layers operate similarly to convolutional layers? This work provides evidence that attention layers can perform convolution and, indeed, they often learn to do so in practice. Specifically, we prove that a multi-head self-attention layer with sufficient number of heads is at least as expressive as any convolutional layer. Our numerical experiments then show that self-attention layers attend to pixel-grid patterns similarly to CNN layers, corroborating our analysis. Our code is publicly available1.  

# 1 INTRODUCTION  

Recent advances in Natural Language Processing (NLP) are largely attributed to the rise of the transformer (Vaswani et al., 2017). Pre-trained to solve an unsupervised task on large corpora of text, transformer-based architectures, such as GPT-2 (Radford et al., 2018), BERT (Devlin et al., 2018) and Transformer-XL (Dai et al., 2019), seem to possess the capacity to learn the underlying structure of text and, as a consequence, to learn representations that generalize across tasks. The key difference between transformers and previous methods, such as recurrent neural networks (Hochreiter & Schmidhuber, 1997) and convolutional neural networks (CNN), is that the former can simultaneously attend to every word of their input sequence. This is made possible thanks to the attention mechanism-originally introduced in Neural Machine Translation to better handle long-range dependencies (Bahdanau et al., 2015). With self-attention in particular, the similarity of two words in a sequence is captured by an attention score measuring the distance of their representations. The representation of each word is then updated based on those words whose attention score is highest.  

Inspired by its capacity to learn meaningful inter-dependencies between words, researchers have recently considered utilizing self-attention in vision tasks. Self-attention was first added to CNN by either using channel-based attention (Hu et al., 2018) or non-local relationships across the image (Wang et al., 2018). More recently, Bello et al. (2019) augmented CNNs by replacing some convolutional layers with self-attention layers, leading to improvements on image classification and object detection tasks. Interestingly, Ramachandran et al. (2019) noticed that, even though state-of-the art results are reached when attention and convolutional features are combined, under same computation and model size constraints, self-attention-only architectures also reach competitive image classification accuracy.  

These findings raise the question, do self-attention layers process images in a similar manner to convolutional layers? From a theoretical perspective, one could argue that transfomers have the capacity to simulate any function-including a CNN. Indeed, Pérez et al. (2019) showed that a multilayer attention-based architecture with additive positional encodings is Turing complete under some strong theoretical assumptions, such as unbounded precision arithmetic. Unfortunately, universality results do not reveal how a machine solves a task, only that it has the capacity to do so. Thus, the question of how self-attention layers actually process images remains open.  

Contributions. β In this work, we put forth theoretical and empirical evidence that self-attention layers can (and do) learn to behave similar to convolutional layers:  

1. From a theoretical perspective, we provide a constructive proof showing that self-attention layers can express any convolutional layers.  

Specifically, we show that a single multi-head self-attention layer using relative positional encoding can be re-parametrized to express any convolutional layer.  

I1. Our experiments show that the first few layers of attention-only architectures (Ramachandran et al., 2019) do learn to attend on grid-like pattern around each query pixel, similar to our theoretical construction.  

Strikingly, this behavior is confirmed both for our quadratic encoding, but also for relative encoding that is learned. Our results seem to suggest that localized convolution is the right inductive bias for the first few layers of an image classifying network. We provide an interactive website2 to explore how self-attention exploits localized position-based attention in lower layers and contentbased attention in deeper layers. For reproducibility purposes, our code is publicly available.  

# 2  BACKGROUND ON ATTENTION MECHANISMS FOR VISION  

We here recallthe mathematical formulation of self-attention layers and emphasize the role of positional encodings.  

# 2.1  THE MULTI-HEAD SELF-ATTENTION LAYER  

Let $X\,\in\,\mathbb{R}^{T\times D_{i n}}$ be an input matrix consisting of $T$ tokens in of $D_{i n}$ dimensions each. While in NLP each token corresponds to a word in a sentence, the same formalism can be applied to any sequence of $T$ discrete objects, e.g. pixels. A self-attention layer maps any query token $t\,\in\,[T]$ from $D_{i n}$ to $D_{o u t}$ dimensions as follows:  

$$
\mathrm{Self.Attention}(X)_{t,:}:=\mathrm{softmax}\left(A_{t,:}\right)X W_{\nu a l},
$$  

where we refer to the elements of the $T\times T$ matrix  

$$
\pmb{A}:=\pmb{X}\pmb{W}_{q r y}\pmb{W}_{k e y}^{\top}\pmb{X}^{\top}
$$  

as attention scores and the softmax output3 as attention probabilities. The layer is parametrized by a query matrix $W_{q r y}\;\in\;\mathbb{R}^{D_{i n}\times D_{k}}$ , a key matrix $\boldsymbol{W_{k e y}}^{\prime}\in\;\mathbb{R}^{D_{i n}\times D_{k}}$ and a value matrix $W_{\nu a l}\,\in$ $\mathbb{R}^{D_{i n}\times D_{o u t}}$ .For simplity, we exclude any residual connections, batch normalization and constant factors.  

A key property of the self-attention model described above is that it is equivariant to reordering, that is, it gives the same output independently of how the $T$ input tokens are shuffled. This is problematic for cases we expect the order of things to matter. To alleviate the limitation, a positional encoding is learned for each token in the sequence (or pixel in an image), and added to the representation of the token itself before applying self-attention  

$$
\pmb{A}:=(\pmb{X}+\pmb{P})\mathbf{W}_{q r y}\mathbf{W}_{k e y}^{\top}(\pmb{X}+\pmb{P})^{\top},
$$  

Where $P\in\mathbb{R}^{T\times D_{i n}}$ contains the embedding vectors for each position. More generally, $_{P}$ may be substituted by any function that returns a vector representation of the position.  

It has been found beneficial in practice to replicate this self-attention mechanism into multiple heads, each being_able to focus on different parts of the input by using different query, key and value matrices. In multi-head self-attention, the output of the $N_{h}$ heads of output dimension $D_{h}$ are concatenated and projected to dimension $D_{o u t}$ asfollows:  

$$
\mathrm{MHSA}(X):=\operatorname{concat}_{h\in[N_{h}]}\left[\mathrm{Self-Attention}_{h}(X)\right]W_{o u t}\,+b_{o u t}
$$  

and two new parameters are introduced: the projection matrix $W_{o u t}\in\mathbb{R}^{N_{h}D_{h}\times D_{o u t}}$ and a bias term bout E IRDou.  

Convolutional layers are the de facto choice for building neural networks that operate on images. Werecalltatgveanmae $\pmb{\Upsilon}\in\mathbb{R}^{W\times H\times D_{i n}}$ of width $W$ ,height $H$ and $D_{i n}$ channels, the output of a convolutional layer for pixel $(i,j)$ is given by  

$$
\mathrm{Conv}(X)_{i,j,:}:=\sum_{(\delta_{1},\delta_{2})\in\Delta_{K}}\mathsf{X}_{i+\delta_{1},j+\delta_{2},:}\mathsf{W}_{\delta_{1},\delta_{2},:,:}+b,
$$  

where $\boldsymbol{\mathsf{W}}$ is the $K\times K\times D_{i n}\times D_{o u t}$ weight tensor 4, $\pmb{b}\in\mathbb{R}^{D_{o u t}}$ is the bias vector and the set  

$$
\mathbb{A}_{K}:=\left[-\left\lfloor{\frac{K}{2}}\right\rfloor,\cdot\cdot\cdot\,,\left\lfloor{\frac{K}{2}}\right\rfloor\right]\times\left[-\left\lfloor{\frac{K}{2}}\right\rfloor,\cdot\cdot\cdot\,,\left\lfloor{\frac{K}{2}}\right\rfloor\right]
$$  

contains all possible shifts appearing when convolving the image with a $K\times K$ kernel.  

In the following, we review how self-attention can be adapted from 1D sequences to images.  

With images, rather than tokens, we have query and key pixels $q,k\in[W]\times[H]$ .Accordingly, the input is a tensor $\pmb{\Upsilon}$ of dimension $W\times H\times D_{i n}$ and each attention score associates a query and a key pixel.  

To keep the formulas consistent with the 1D case, we abuse notation and slice tensors by using a 2D index vector: if $\pmb{p}=(i,j)$ : we write $\pmb{\ X_{p}},$ : and $\mathsf{A}_{p,}$ : to mean $\pmb{\ X}_{i,j},$ : and $\pmb{\mathsf{A}}_{i,j,:,}$ , respectively. With this notation in place, the multi-head self attention layer output at pixel $\pmb q$ can be expressed as follows:  

$$
\mathrm{Self-Attention}(\pmb{X})_{\pmb{q},:}=\sum_{k}\mathrm{softmax}\left(\pmb{\mathsf{A}}_{\pmb{q},:}\right)_{k}\pmb{\mathsf{X}}_{k,:}\,W_{\nu a l}
$$  

and accordingly for the multi-head case.  

# 2.3POSITIONAL ENCODING FOR IMAGES  

There are two types of positional encoding that has been used in transformer-based architectures: the absolute and relative encoding (see also Table 3 in the Appendix).  

With absolute encodings, a (fixed or learned) vector $\mathsf{P}_{p,}$ : is assigned to each pixel $\textbf{\emph{p}}$ Thecomputation of the attention scores we saw in eq. (2) can then be decomposed as follows:  

$$
\begin{array}{r l}&{\mathsf{A}_{q,k}^{\mathrm{abs}}=(\mathbf{X}_{q,:}+\mathsf{P}_{q,:})W_{q r y}W_{k e y}^{\top}(\mathbf{X}_{k,:}+\mathsf{P}_{k,:})^{\top}}\\ &{\qquad=\mathbf{X}_{q,:}W_{q r y}W_{k e y}^{\top}\mathbf{X}_{k,:}^{\top}+\mathbf{X}_{q,:}W_{q r y}W_{k e y}^{\top}\mathsf{P}_{k,:}^{\top}+\mathsf{P}_{q,:}W_{q r y}W_{k e y}^{\top}\mathbf{X}_{k,:}+\mathsf{P}_{q,:}W_{q r y}W_{k e y}^{\top}\mathsf{P}_{k,:}}\end{array}
$$  

where $\pmb q$ and $^k$ correspond to the query and key pixels, respectively.  

The relative positional encoding was introduced by Dai et al. (2019). The main idea is to only consider the position difference between the query pixel (pixel we compute the representation of) and the key pixel (pixel we attend) instead of the absolute position of the key pixel:  

$$
\begin{array}{r}{\mathsf{A}_{q,k}^{\mathrm{rel}}:=\mathsf{X}_{q,:}^{\top}W_{q r y}^{\top}W_{k e y}\,\mathsf{X}_{k,:}+\mathsf{X}_{q,:}^{\top}W_{q r y}^{\top}\widehat{W}_{k e y}\,r_{\delta}+u^{\top}W_{k e y}\,\mathsf{X}_{k,:}+v^{\top}\widehat{W}_{k e y}\,r_{\delta}}\end{array}
$$  

In this manner, the attention scores only depend on the shift $\delta\;:=\;k\mathrm{~-~}q$ .Above, the learnable vectors $\textbf{\em u}$ and $\pmb{v}$ are unique for each head, whereas for every shift $\delta$ the relative positional encoding $r_{\delta}\in\mathbb{R}^{D_{p}}$ is shared by all layers and heads. Moreover, now the key weights are split into two types: $W_{k e y}$ pertain to the input and $\widehat{W}_{k e y}$ to the relative position of pixels.  

# 3 SELF-ATTENTION AS A CONVOLUTIONAL LAYER  

This section derives sufficient conditions such that a multi-head self-attention layer can simulate a convolutional layer. Our main result is the following:  

Theorem 1. A multi-head self-attention layer with $N_{h}$ heads of dimension $D_{h}$ ,output dimension $D_{o u t}$ and a relative positional encoding of dimension $D_{p}\,\geq\,3$ can express any convolutional layer of kernel size $\sqrt{N_{h}}\times\sqrt{N_{h}}$ and $\operatorname*{min}(D_{h},D_{o u t})$ output channels.  

The theorem is proven constructively by selecting the parameters of the multi-head self-attention layer so that the latter acts like a convolutional layer. In the proposed construction, the attention scores of each self-attention head should attend to a different relative shift within the set $\mathbb{\Delta}_{K}\,=$ $\{-\lfloor K/2\rfloor,\ldots,\lfloor K/2\rfloor\}^{2}$ of all pixel shifts in a $K\times K$ kernel. The exact condition can be found in the statement of Lemma 1.  

Then, Lemma 2 shows that the aforementioned condition is satisfied for the relative positional encoding that we refer to as the quadratic encoding:  

$$
\begin{array}{r}{v^{(h)}:=-\alpha^{(h)}\left(1,-2\Delta_{1}^{(h)},-2\Delta_{2}^{(h)}\right)\quad r_{\delta}:=\left(\|\delta\|^{2},\delta_{1},\delta_{2}\right)\quad W_{q r y}:=W_{k e y}:=\mathbf{0}\quad\widehat{W_{k e y}}:=I}\end{array}
$$  

The learned parameters $\pmb{\Delta}^{(h)}=(\pmb{\Delta}_{1}^{(h)},\pmb{\Delta}_{2}^{(h)})$ and $\alpha^{(h)}$ determine the center and width of attention of each head, respectively. On the other hand, $\pmb{\delta}=(\pmb{\delta}_{1},\pmb{\delta}_{2})$ is fixed and expresses the relative shift between query and key pixels.  

It is important to stress that the above encoding is not the only one for which the conditions of Lemma 1 are satisfied. In fact, in our experiments, the relative encoding learned by the neural network also matched the conditions of the lemma (despite being different from the quadratic encoding). Nevertheless, the encoding defined above is very efficient in terms of size, as only $D_{p}=3$ dimensions suffice to encode the relative position of pixels, while also reaching similar or better empirical performance (than the learned one).  

The theorem covers the general convolution operator as defined in eq. (17). However, machine learning practitioners using differential programming frameworks (Paszke et al., 2017; Abadi et al., 2015) might question if the theorem holds for all hyper-parameters of 2D convolutional layers:  

· Padding: a multi-head self-attention layer uses by default the " SAME" padding while a convolutional layer would decrease the image size by $K-1$ pixels. The correct way to alleviate these boundary effects is to pad the input image with $\lfloor K/2\rfloor$ zeros on each side. In this case, the cropped output of a MHSA and a convolutional layer are the same. · Stride: a strided convolution can be seen as a convolution followed by a fixed pooling operation—with computational optimizations. Theorem 1 is defined for stride 1, but a fixed pooling layer could be appended to the Self-Attention layer to simulate any stride. · Dilation: a multi-head self-attention layer can express any dilated convolution as each head can attend a value at any pixel shift and form a (dilated) grid pattern.  

Remark for the ID case.  Convolutional layers acting on sequences are commonly used in the literature for text (Kim, 2014), as well as audio (van den Oord et al., 2016) and time series (Franceschi et al., 2019). Theorem 1 can be straightforwardly extended to show that multi-head self-attention With $N_{h}$ heads can also simulate a 1D convolutional layer with a kernel of size ${\cal K}\;=\;{\cal N}_{h}$ With $\operatorname*{min}(D_{h},D_{o u t})$ output channels using a positional encoding of dimension $D_{p}\geq2$ .Sincewehave not tested empirically if the preceding construction matches the behavior of 1D self-attention in practice, we cannot claim that it actually learns to convolve an input sequence—-only that it has the capacitytodoso.  

PROOF OF MAIN THEOREM  

The proof follows directly from Lemmas 1 and 2 stated below:  

Lemma 1. Consider a multi-head self-attention layer consisting of $N_{h}\,=\,K^{2}$ heads, $D_{h}\,\geq\,D_{o u t}$ andlet $f~:~[N_{h}]~\rightarrow~\Delta_{K}$ be a bijective mapping of heads onto shifts. Further, suppose that for every head thefollowingholds:  

$$
\mathrm{softmax}(A_{\pmb{q},:}^{(h)})_{k}=\left\{1\atop0\quad o t h e r w i s e.\right.
$$  

Then, for any convolutional layer with a $K\times K$ kernel and $D_{o u t}$ output channels, there exists $\{W_{\nu a l}^{(h)}\}_{h\in[N_{h}]}$ suchat $\mathrm{MHSA}(X)=\mathrm{Conv}(X)$ forevery $\pmb{X}\in\mathbb{R}^{W\times H\times D_{i n}}$  

![](images/0e4402e9e07d166170072610182aecda5275873e68c255f08e9315959083006e.jpg)  

Figure 1: Ilustration of a Multi-Head Self-Attention layer applied to a tensor image $\pmb{\ X}.$ Each head $h$ attends pixel values around shift △(h) and learn a flter matrix W() . We show attention maps computed for a query pixel at position $\pmb q$  

Proof. Our first step will be to rework the expression of the Multi-Head Self-Attention operator from equation (1) and equation (4) such that the effect of the multiple heads becomes more transparent:  

$$
\mathrm{MHSA}(X)=b_{o u t}+\sum_{h\in[N_{h}]}\mathrm{softmax}(A^{(h)})X\underbrace{W_{v a l}^{(h)}W_{o u t}[(h-1)D_{h}+1:h D_{h}+1]}_{W^{(h)}}
$$  

$W_{\nu a l}^{(h)}\in\mathbb{R}^{D_{i n}\times D_{h}}$ $W_{o u t}$ $D_{h}\,\times\,D_{o u t}$ $D_{h}\ \geq\ D_{o u t}$ matrices by a learned matrix $W^{(h)}$ for each head. We consider one output pixel of the multi-head self-attention:  

$$
\mathrm{MHSA}(\boldsymbol{X})_{q,:}=\sum_{h\in[N_{h}]}\left(\sum_{k}\mathrm{softmax}(\mathbb{A}_{q,:}^{(h)})_{k}\pmb{\mathsf{X}}_{k,:}\right)\boldsymbol{W}^{(h)}+b_{o u t}
$$  

Due to the conditions of the Lemma, for the $h$ -th attention head the attention probability is one when $\pmb{k}=\pmb{q}-\pmb{f}(h)$ and zero otherwise. The layer's output at pixel $\pmb q$ is thus equal to  

$$
\mathrm{MHSA}(\pmb{\mathsf{X}})_{\pmb{q}}=\sum_{\pmb{h}\in[N_{h}]}\pmb{\mathsf{X}}_{\pmb{q}-\pmb{f}(h),:}W^{(h)}+\pmb{b}_{o u t}
$$  

For $K=\sqrt{N_{h}}$ , the above can be seen to be equivalent to a convolutional layer expressed in eq. 17: there is a one to one mapping (implied by map $\pmb{f}$ between thematrices $\boldsymbol{W}^{(\dot{h})}$ for $h=[N_{h}]$ and the matrices $\mathsf{W}_{k_{1},k_{2},:,:}$ for all $(\bar{k_{1}},\bar{k}_{2})^{2}\in[K]^{2}$ 口  

Remark  about $D_{h}$ and $D_{o u t}$ . It is frequent in  transformer-based  architectures to  set $D_{h}~=~D_{o u t}/N_{h}$ , hence $D_{h}\,<\,D_{o u t}$ . In that case, $W^{(h)}$ can be seen to be of rank $D_{o u t}-D_{h}$ ， which does not suffice to express every convolutional layer with $D_{o u t}$ channels. Nevertheless, it can be seen that any $D_{h}$ out of $D_{o u t}$ outputs of $\mathrm{MHSA}(X)$ can express the output of any convolutional layer with $D_{h}$ output channels. To cover both cases, in the statement of the main theorem we assert that the output channels of the convolutional layer should be $\operatorname*{min}(D_{h},D_{o u t})$ . In practice, we advise to concatenate heads of dimension $D_{h}=D_{o u t}$ instead of splitting the $D_{o u t}$ dimensions among heads to have exact re-parametrization and no “unused" channels.  

Lemma 2. There exists a relative encoding scheme $\{r_{\delta}\,\in\,\mathbb{R}^{D_{p}}\}_{\delta\in\mathbb{Z}^{2}}$ with $D_{p}\,\geq\,3$ and parameters $W_{q r y}$ ， $W_{k e y}$ ， $\widehat{W}_{k e y},\boldsymbol{u}$ with $D_{p}\,\leq\,D_{k}$ such that, for every $\pmb{\Delta}\in\mathbb{A}_{K}$ there exists some vector $\pmb{v}$ (conditioned on $\Delta$ ) yielding softm $\operatorname{tax}(\mathbf{A}_{q,:})_{k}=1\;i f k-q=\Delta$ and zero, otherwise.  

Proof. We show by construction the existence of a $D_{p}=3$ dimensional relative encoding scheme yielding the required attention probabilities.  

As the attention probabilities are independent of the input tensor $\pmb{\Upsilon}$ we set $W_{k e y}=W_{q r y}=\mathbf{0}$ which leaves only the last term of eq. (8). Setting $\widehat{W}_{k e y}\in\mathbb{R}^{D_{k}\times D_{p}}$ to the identity matrix (with appropriate row padding), yields $\pmb{\mathsf{A}}_{\pmb{q},\pmb{k}}=\pmb{v}^{\top}\pmb{r}_{\delta}$ where $\pmb{\delta}:=\pmb{k}-\pmb{q}$ . Above, we have assumed that $D_{p}\leq D_{k}$ such that no information from $r_{\delta}$ is lost.  

Now, suppose that we could write:  

$$
\mathbf{\deltaA}_{q,k}=-\alpha(||\delta-\Delta||^{2}+c)
$$  

for some constant $c$ . In the above expression, the maximum attention score over $\mathsf{A}_{q,}$ : is $-\alpha c$ and it is reached for $\mathsf{A}_{q,k}$ with $\pmb\delta=\Delta$ . On the other hand, the $\alpha$ coefficient can be used to scale arbitrarily the difference between $\mathsf{A}_{q,\Delta}$ and the other attention scores.  

In this way, for $\pmb\delta=\Delta$ , we have  

$$
\begin{array}{l}{\displaystyle\operatorname*{lim}_{\alpha\to\infty}\mathrm{softmax}(\mathsf{A}_{q,:})_{k}=\operatorname*{lim}_{\alpha\to\infty}\frac{e^{-\alpha(\|\delta-\Delta\|^{2}+c)}}{\sum_{k^{\prime}}e^{-\alpha(\|(k-q^{\prime})-\Delta\|^{2}+c)}}\qquad\qquad\qquad\qquad\qquad\qquad}\\ {\displaystyle=\operatorname*{lim}_{\alpha\to\infty}\frac{e^{-\alpha\|\delta-\Delta\|^{2}}}{\sum_{k^{\prime}}e^{-\alpha\|(k-q^{\prime})-\Delta\|^{2}}}=\frac{1}{1+\operatorname*{lim}_{\alpha\to\infty}\sum_{k^{\prime}\neq k}e^{-\alpha\|(k-q^{\prime})-\Delta\|^{2}}}=1}\end{array}
$$  

and for $\delta\neq\Delta$ , the equation becomes $\mathrm{lim}_{\alpha\to\infty}$ softmax $(\pmb{\mathsf{A}}_{q,:})_{k}=0$ , exactly as needed to satisfy the lemma statement.  

What remains is to prove that there exist $\pmb{v}$ and $\{r_{\delta}\}_{\delta\in\mathbb{Z}^{2}}$ for which eq. (14) holds. Expanding the RHS of the equation, we have $-\alpha(\|\pmb\delta-\Delta\|^{2}+\pmb\ c)=-\alpha(\|\pmb\delta\|^{2}+\|\Delta\|^{2}-2\langle\pmb\delta,\Delta\rangle+c)$ . Now if we set $\pmb{v}=-\alpha\left(1,-2\pmb{\Delta}_{1},-2\pmb{\Delta}_{2}\right)$ and ${\pmb r}_{\delta}=(\|\pmb\delta\|^{2},\pmb\delta_{1},\pmb\delta_{2})$ , then  

$$
\mathtt{A}_{q,k}=v^{\top}r_{\delta}=-\alpha(\|\delta\|^{2}-2\Delta_{1}\delta_{1}-2\Delta_{2}\delta_{2})=-\alpha(\|\delta\|^{2}-2\langle\delta,\Delta\rangle)=-\alpha(\|\delta-\Delta\|^{2}-\|\Delta\|^{2}),
$$  

which matches eq. (14) with $c=-\|\pmb{\Delta}\|^{2}$ and the proof is concluded.  

Remark on the magnitude of $\alpha$ .  The exact representation of one pixel requires $\alpha$ (or the matrices $W_{q r y}$ and $W_{k e y,}$ 0 to be arbitrary large, despite the fact that the attention probabilities of all other pixels converge exponentially to O as $\alpha$ grows. Nevertheless, practical implementations always rely on finite precision arithmetic for which a constant $\alpha$ suffices to satisfy our construction. For instance, since the smallest positive f loat 32 scalar is approximately $10^{-45}$ , setting $\alpha=46$ would suffice to obtain hard attention.  

# 4 EXPERIMENTS  

The aim of this section is to validate the applicability of our theoretical results-which state that self-attention can perform convolution—and to examine whether self-attention layers in practice do actually learn to operate like convolutional layers when trained on standard image classification tasks. In particular, we study the relationship between self-attention and convolution with quadratic and learned relative positional encodings. We find that, for both cases, the attention probabilities learned tend to respect the conditions of Lemma 1, supporting our hypothesis.  

# 4.1 IMPLEMENTATION DETAILS  

We study a fully attentional model consisting of six multi-head self-attention layers. As it has already been shown by Bello et al. (2019) that combining attention features with convolutional features improves performance on Cifar-100 and ImageNet, we do not focus on attaining state-of-the-art performance. Nevertheless, to validate that our model learns a meaningful classifier, we compare it to the standard ResNet18 (He et al., 2015) on the CIFAR-10 dataset (Krizhevsky et al.). In all experiments,we use a $2\times2$ invertible down-sampling (Jacobsen et al., 2018) on the input to reduce the size of the image. As the size of the attention coefficient tensors (stored during forward) scales quadratically with the size of the input image, full attention cannot be applied to bigger images. The fixed size representation of the input image is computed as the average pooling of the last layer representations and given to a linear classifier.  

![](images/f1dddba6517a85f9752015e8353614ff69cbe35de90d7de82667634558c07713.jpg)  
Figure 2: Test accuracy on CIFAR-10.  

![](images/7e06e79f022cd0307b58e5c3dc2ecb0434a22565390f4c647bea09e542d56a14.jpg)  
Table 1: Test accuracy on CIFAR-10 and model sizes. SA stands for Self-Attention.  

![](images/0de897fe806de5263bd46d4cac74d935a93600323c728ef29e144c313c8490f0.jpg)  

Figure 3: Centers of attention of each attention head (different colors) at layer 4 during the training with quadratic relative positional encoding. The central black square is the query pixel, whereas solid and dotted circles represent the $50\%$ and $90\%$ percentiles of each Gaussian, respectively.  

We used the PyTorch library (Paszke et al., 2017) and based our implementation on PyTorch Transformers?. We release our code on Github6 and hyper-parameters are listed in Table 2 (Appendix).  

Remark on accuracy.  To verify that our self-attention models perform reasonably well, we display in Figure 6 the evolution of the test accuracy on CIFAR-10 over the 300 epochs of training for our self-attention models against a small ResNet (Table 1). The ResNet is faster to converge, but we cannot ascertain whether this corresponds to an inherent property of the architecture or an artifact of the adopted optimization procedures. Our implementation could be optimized to exploit the locality of Gaussian attention probabilities and reduce significantly the number of FLOPS. We observed that learned embeddings with content-based attention were harder to train probably due to their increased number of parameters. We believe that the performance gap can be bridged to match the ResNet performance, but this is not the focus of this work.  

# 4.2 QUADRATIC ENCODING  

As a first step, we aim to verify that, with the relative position encoding introduced in equation (9), attention layers learn to behave like convolutional layers. We train nine attention heads at each layer tobeonparwiththe $3\times3$ kernels used predominantly by the ResNet architecture. The center of attention of each head $h$ is initialized to $\bar{\Delta^{(h)}}\sim\mathcal{N}(\mathbf{0},\bar{2}I_{2})$  

Figure 3 shows how the initial positions of the heads (different colors) at layer 4 changed during training. We can see that after optimization, the heads attend on specific pixel of the image forming a grid around the query pixel. Our intuition that Self-Attention applied to images learns convolutional filters around the queried pixel is confirmed.  

Figure 4 displays all attention head at each layer of the model at the end of the training. It can be seen that in the first few layers the heads tend to focus on local patterns (layers 1 and 2), while deeper layers (layers 3-6) also attend to larger patterns by positioning the center of attention further from the queried pixel position. We also include in the Appendix a plot of the attention positions for a higher number of heads ( $N_{h}=16)$ . Figure 14 displays both local patterns similar to CNN and long range dependencies. Interestingly, attention heads do not overlap and seem to take an arrangement maximizing the coverage of the input space.  

![](images/a06fde5f1516da947e4f518ef1898220f67fa98a3f00672d99bfca19ac030bd0.jpg)  

Figure 4: Centers of attention of each attention head (different colors) for the 6 self-attention layers using quadratic positional encoding. The central black square is the query pixel, whereas solid and dotted circles represent the $50\%$ and $90\%$ percentiles of each Gaussian, respectively.  

# 4.3LEARNED RELATIVE POSITIONAL ENCODING  

We move on to study the positional encoding used in practice by fully-attentional models on images.  

We implemented the 2D relative positional encoding scheme used by (Ramachandran et al., 2019; Bello et al., 2019): we learn a $\bar{\lfloor D_{p}\bar{/}2\rfloor}$ position encoding vector for each row and each column pixel shift. Hence, the relative positional encoding of a key pixel at position $^k$ with a query pixel at position $\pmb q$ is the concatenation of the row shift embedding $\delta_{1}$ and the column shift embedding $\delta_{2}$ (where $\pmb{\delta}=\pmb{k}-\pmb{q})$ : We chose $D_{p}\,=\,D_{o u t}\,=\,400$ in the experiment. We differ from their (unpublished) implementation in the following points: $(i)$ we do not use convolution stem and ResNet bottlenecks for downsampling, but only a $2\times2$ invertible downsampling layer (Jacobsen et al., 2018) at input, $(i i)$ we use $D_{h}=D_{o u t}$ instead of $D_{h}=D_{o u t}/N_{h}$ backed by our theory that the effective number of learned filters is $\operatorname*{min}(D_{h},D_{o u t})$  

At first, we discard the input data and compute the attention scores solely as the last term of eq. (8). The attention probabilities of each head at each layer are displayed on Figure 5. The figure confirms our hypothesis for the first two layers and partially for the third: even when left to learn the positional encoding scheme from randomly initialized vectors, certain self-attention heads (depicted on the left) learn to attend to individual pixels, closely matching the condition of Lemma 1 and thus Theorem 1. At the same time, other heads pay attention to horizontally-symmetric but non-localized patterns, as well as to long-range pixel inter-dependencies.  

We move on to a more realistic setting where the attention scores are computed using both positional and content-based attention (i.e., $q^{\top}{\bar{k}}+q^{\top}r$ in (Ramachandran et al., 2019)) which corresponds to a full-blown standalone self-attention model.  

The attention probabilities of each head at each layer are displayed in Figure 6. We average the attention probabilities over a batch of 100 test images to outline the focus of each head and remove the dependency on the input image. Our hypothesis is confirmed for some heads of layer 2 and 3: even when left to learn the encoding from the data, certain self-attention heads only exploit positionbased attention to attend to distinct pixels at a fixed shift from the query pixel reproducing the receptive field of a convolutional kernel. Other heads use more content-based attention (see Figures 8 to 10 in Appendix for non-averaged probabilities) leveraging the advantage of Self-Attention over CNN which does not contradict our theory. In practice, it was shown by Bello et al. (2019) that combining CNN and self-attention features outperforms each taken separately. Our experiments shows that such combination is learned when optimizing an unconstrained fully-attentional model.  

The similarity between convolution and multi-head self-attention is striking when the query pixel is slid over the image: the localized attention patterns visible in Figure 6 follow the query pixel. This characteristic behavior materializes when comparing Figure 6 with the attention probabilities at a different query pixel (see Figure 7 in Appendix). Attention patterns in layers 2 and 3 are not only localized but stand at a constant shift from the query pixel, similarly to convolving the receptive field of a convolutional kernel over an image. This phenomenon is made evident on our interactive website?. This tool is designed to explore different components of attention for diverse images with or without content-based attention. We believe that it is a useful instrument to further understand how MHSA learns to process images.  

![](images/01cbe7ba9d3e9ea6a9ecae1a80b648620396bf8504d52dc063be25d31e7dc0d1.jpg)  
Figure 5: Attention probabilities of each head (column) at each layer $(r o w)$ using learned relative positional encoding without content-based attention. The central black square is the query pixel. We reordered the heads for visualization and zoomed on the $7\mathrm{x}7$ pixels around the query pixel.  

![](images/d55b61a35d89e5a5d4d59265d49162c90c8bb5cd47394b30b1fc8c345cdc655a.jpg)  
Figure 6: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content based attention. Attention maps are averaged over 1o0 test images to display head behavior and remove the dependence on the input content. The black square is the query pixel. More examples are presented in Appendix A.  

# 5 RELATED WORK  

In this section, we review the known differences and similarities between CNNs and transformers.  

The use of CNN networks for text-at word level (Gehring et al., 2017) or character level (Kim, 2014)is more seldom than transformers (or RNN). Transformers and convolutional models have been extensively compared empirically on tasks of Natural Language Processing and Neural Machine Translation. It was observed that transformers have a competitive advantage over convolutional model applied to text (Vaswani et al., 2017). It is only recently that Bello et al. (2019); Ramachandran et al. (2019) used transformers on images and showed that they achieve similar accuracy as ResNets. However, their comparison only covers performance and number of parameters and FLOPS but not expressive power.  

Beyond performance and computational-cost comparisons of transformers and CNN, the study of expressiveness of these architectures has focused on their ability to capture long-term dependencies (Dai et al., 2019). Another interesting line of research has demonstrated that transformers are Turingcomplete (Dehghani et al., 2018; Pérez et al., 2019), which is an important theoretical result but is not informative for practitioners. To the best of our knowledge, we are the first to show that the class of functions expressed by a layer of self-attention encloses all convolutional filters.  

The closest work in bridging the gap between attention and convolution is due to Andreoli (2019). They cast attention and convolution into a unified framework leveraging tensor outerproduct. In this framework, the receptive field of a convolution is represented by a “basis"” tensor $\mathbf{\dot{A}}\ \in\ \mathbb{R}^{K\times K\times H\times W\times H\times W}$ Fornstrifl $K\times K$ convolutional kernel would be encoded by $\mathbf{A}_{\Delta,q,k}\,=\,\mathbb{1}\{k\,-\,q\,\bar{=}\,\Delta\}$ for $\pmb{\Delta}\in\mathbb{A}_{K}$ . The author distinguishes this index-based convolution with content-based convolution where $\pmb{\triangle}$ is computed from the value of the input, e.g., using a key/query dot-product attention. Our work moves further and presents sufficient conditions for relative positional encoding injected into the input content (as done in practice) to allow content-based convolution to express any index-based convolution. We further show experimentally that such behavior is learned in practice.  

# 6 CONCLUSION  

We showed that self-attention layers applied to images can express any convolutional layer (given sufficiently many heads) and that fully-attentional models learn to combine local behavior (similar to convolution) and global attention based on input content. More generally, fully-attentional models seem to learn a generalization of CNNs where the kernel pattern is learned at the same time as the filters—similar to deformable convolutions (Dai et al., 2017; Zampieri, 2019). Interesting directions for future work include translating existing insights from the rich CNNs literature back to transformers on various data modalities, including images, text and time series.  

# ACKNOWLEDGMENTS  

Jean-Baptiste Cordonnier is thankful to the Swiss Data Science Center (SDsC) for funding this work. Andreas Loukas was supported by the Swiss National Science Foundation (project “Deep Learning for Graph Structured Data", grant number PZ00P2 179981).  

# REFERENCES  

Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.   
Jean-Marc Andreoli. Convolution, attention and structure embedding. NeurIPS 2019 workshop on Graph Representation Learning, Dec 13, 2019, Vancouver, BC, Canada, 2019.   
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.   
Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V. Le. Attention Augmented Convolutional Networks. arXiv:1904.09925 [cs], April 2019.   
Jifeng Dai, Haozhi Qi, Yuwen Xiong, YiLi, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. CoRR, abs/1703.06211, 2017.   
Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. CoRR, abs/1901.02860, 2019.   
Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. CoRR, abs/1807.03819, 2018.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805, 2018.   
Jean-Yves Franceschi, Aymeric Dieuleveut, and Martin Jagi. Unsupervised scalable representation learning for multivariate time series. In NeurIPS 2019, 2019.   
Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. CoRR, abs/1705.03122, 2017.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.   
Sepp Hochreiter and Juirgen Schmidhuber. Long short-term memory. Neural Computation, 9(8): 1735-1780,1997.   
Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, Pp. 7132-7141, 2018.   
Jorn-Henrik Jacobsen, Arnold W.M. Smeulders, and Edouard Oyallon. i-revnet: Deep invertible networks. In International Conference on Learning Representations, 2018.   
Yoon Kim.  Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1746- 1751, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/ D14-1181.   
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 (canadian institute for advanced research).   
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. In NIPS Autodiff Workshop, 2017.   
Jorge Perez, Javier Marinkovic, and Pablo Barcel6. On the turing completeness of modern neural network architectures. CoRR, abs/1901.03429, 2019.   
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and lya Sutskever. Language models are unsupervised multitask learners. 2018.   
Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jonathon Shlens. Stand-alone self-attention in vision models. CoRR, abs/1906.05909, 2019.   
Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alexander Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. CoRR, abs/1706.03762, 2017.   
Xiaolong Wang, Ross B. Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In 2018 iEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pp. 7794-7803, 2018.   
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime G. Carbonell, Ruslan Salakhutdinov, and Quoc V. Le.  Xlnet:  Generalized autoregressive pretraining for language understanding. CoRR, abs/1906.08237, 2019.   
Luca Zampieri. Geometric deep learning for volumetric computational fuid dynamics. Pp. 67, 2019.  

# APPENDIX  

AMORE EXAMPLES WITH CONTENT-BASED ATTENTION  

We present more examples of attention probabilities computed by self-attention model. Figure 7 shows average attention at a different query pixel than Figure 6. Figures 8 to 10 display attention for single images.  

![](images/1463c7f94677fa15ded6d126a8647a856366baaf7ae95ba286fc0291b15502ee.jpg)  
Figure 7: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content attention. We present the average of 100 test images. The black square is the query pixel.  

![](images/e1b680987f80ddeaf9744d262b89b1a99ad2de47ec4b196e1ab81ebcfdf9f7da.jpg)  
Figure 8: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content based attention. The query pixel (black square) is on the frog head.  

![](images/8beaad7f4da6a9d98e29be849b3c772026acd079dcf4e5d5edb49d4bf7b07650.jpg)  
Figure 9: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content based attention. The query pixel (black square) is on the horse head.  

![](images/e1ea487925f01fec00fa2fdb568103549e20852cc3cb99d21efcc228cb1c602d.jpg)  
Figure 10: Attention probabilities for a model with 6 layers (rows) and 9 heads (columns) using learned relative positional encoding and content-content based attention. The query pixel (black square) is on the building in the background.  

# B HYPER-PARAMETERS USED IN OUR EXPERIMENTS  

![](images/ed0301473e6c4e1567b61ce936018e2703c07f2f6cef920d7daa92be36d3ed9f.jpg)  
Table 2: Self-attention network parameters  

C  POSITIONAL ENCODING REFERENCES   

![](images/74a4c515dbdeb724e31570c9c96918ff71d99f14d3dce8dc5d6d737c184466d4.jpg)  
Table 3: Types of positional encoding used by transformers models applied to text (top) and images (bottom). When multiple encoding types have been tried, we report the one advised by the authors.  

# D GENERALIZED LEMMA 1  

We present a generalization of Lemma 1 that replaces the necessity of hard attention (to single pixels) by a milder assumption: the attention probabilities should span the grid receptive field. The conditions of this Lemma are still satisfied by Lemma 2, hence Theorem 1 follows.  

Lemma 3. Consider a multi-head self-attention layer consisting of $N_{h}\geq K^{2}$ heads, $D_{h}\,\geq\,D_{o u t}$ andlet $\omega\;:\;[H]\times[W]\;\rightarrow\;[H W]$ be a pixel indexing. Then, for any convolutional layer with a $K\times$ $K$ kemel and $D_{o u t}$ outpit chamels there xists $\{W_{\nu a l}^{(h)}\}_{h\in[N_{h}]}$ and $W_{o u t}$ such that $\mathrm{MHSA}(\mathbf{X})=$ $\operatorname{Conv}(\mathbf{X})$ for every $\pmb{\mathsf{X}}\in\mathbb{R}^{W\times H\times D_{i n}}$ if and only if for all $\pmb{q}\in[H]\times[W],$  

$$
\begin{array}{r}{\mathrm{span}\big(\{e_{\omega(q+\Delta)}\in\mathbb{R}^{H W}:\Delta\in\Delta_{K}\}\big)\subseteq\mathrm{span}\big(\big\{\mathrm{vect}(\mathrm{softmax}(\pmb{\mathscr{A}}_{q,:}^{(h)})):h\in[N_{h}]\big\}\big)\,.}\end{array}
$$  

![](images/9039e784ab603717086c58dae714082a5f052787bb19319e13e98f6c8288ecd1.jpg)  
Figure 11: Factorization of the vectorized weight matrices $V_{q}^{\mathrm{conv}}$ and $V_{q}^{\mathrm{SA}}$ used to compute the output at position $\pmb q$ for an input image of dimension $H\times W$ . On the left: a convolution of kernel $2\times2$ , on the right: a self-attention with $N_{h}=5$ heads. $D_{i n}=2$ $D_{o u t}=3$ in both cases.  

Proof. Our first step will be to rework the expression of the Multi-Head Self-Attention operator from equation (1) and equation (4) such that the effect of the multiple heads becomes more transparent:  

$$
\mathrm{MHSA}(\pmb{\mathsf{X}})=b_{o u t}+\sum_{h\in[N_{h}]}\mathrm{softmax}(\pmb{\mathsf{A}}^{(h)})\pmb{\mathsf{X}}\underbrace{W_{v a l}^{(h)}W_{o u t}[(h-1)D_{h}+1:h D_{h}+1]}_{W^{(h)}}
$$  

Note that each head's value matrx $W_{\nu a l}^{(h)}\in\mathbb{R}^{D_{i n}\times D_{h}}$ and each block of the projection matrix $W_{o u t}$ of dimension $D_{h}\,\times\,D_{o u t}$ are learned. Assuming that $D_{h}\ \geq\ D_{o u t}$ , we can replace each pair of matrices by a learned matrix $W^{(h)}$ for each head. We consider one output pixel of the multi-head self-attention and drop the bias term for simplicity:  

$$
\mathrm{MHSA}(\pmb{X})_{q;\mathrm{=}}\sum_{h\in[N_{h}]}\Big(\sum_{\pmb{k}}a_{\pmb{q},\pmb{k}}^{(h)}\pmb{\mathsf{X}}_{\pmb{k},\cdot}\Big){W}^{(h)}=\sum_{\pmb{k}}\pmb{\mathsf{X}}_{\pmb{k},\cdot}\underbrace{\Big(\sum_{h\in[N_{h}]}a_{\pmb{q},\pmb{k}}^{(h)}\pmb{W}^{(h)}\Big)}_{W_{\pmb{q},\pmb{k}}^{\mathrm{SA}}\in\mathbb{R}^{D_{h}\times D_{o u t}}}\,,
$$  

With a.g.k $a_{\pmb q,\pmb k}^{(h)}=\mathrm{softmax}(\pmb{\mathsf{A}}_{\pmb q,\pm}^{(h)})_{\pmb k}$ Werewrite theoutput of aconvolution at pixel $\pmb q$ in the same manner:  

$$
\mathrm{Conv}(\mathbf{X})_{q,:}=\sum_{\Delta\in\Delta_{K}}\mathsf{X}_{q+\Delta,:}\mathsf{W}_{\Delta,:,:}=\sum_{k\in[H]\times[W]}\mathsf{X}_{k,:}\underbrace{\mathbb{1}_{\{k-q\in\Delta_{K}\}}\mathsf{W}_{k-q,:,:}}_{W_{q,k}^{\mathrm{cov}}\in\mathbb{R}^{D_{i n}\times D_{o u t}}}\,.
$$  

Equality between equations (16) and (17) holds for any input $\pmb{\Upsilon}$ if and only if the linear transfor$\dot{W}_{q,k}^{\mathrm{conv}}\,=\,W_{q,k}^{\mathrm{SA}}\;\breve{\forall}q,k$ y,AVq, k. We vectorize the $D_{i n}D_{o u t}\times H W$ $V_{q}^{\mathrm{conv}}:=[\mathrm{vect}(W_{q,k}^{\mathrm{conv}})]_{k\in[H]\times[W]}$   
and $V_{q}^{\mathrm{SA}}:=\,\left[\mathrm{vect}(W_{q,k}^{\mathrm{SA}})\right]_{k\in[H]\times[W]}$ Hence, to show that $\operatorname{Conv}(\mathbf{X})\,=\,\operatorname{MHSA}(\mathbf{X})$ forall $\pmb{\Upsilon}$ we   
must show that $V_{q}^{\mathrm{conv}}=V_{q}^{\mathrm{SA}}$ for ll $\pmb q$  

The matrix $V_{q}^{\mathrm{conv}}$ has a restricted support: only the columns associated with a pixel shift $\pmb{\Delta}\in\mathbb{A}_{K}$ in the recetive field of pixel $\pmb q$ can be non-zero. This eads to the factorization $V_{q}^{\mathrm{conv}}=W^{\mathrm{conv}}E_{q}$ displayed in Figure 11 where $W^{\mathrm{conv}}\,\in\,\mathbb{R}^{D_{i n}D_{o u t}\times K^{2}}$ and $E_{q}\,\in\,\mathbb{R}^{K^{2}\times H W}$ . Given an ordering of the shifts $\pmb{\Delta}\in\mathbb{A}_{K}$ indexed by $j$ , set $(W^{\mathrm{conv}})_{:,j}\,=\,\mathrm{vect}(\bar{\mathbf{W}_{\Delta,:,:}})$ and $(E_{q})_{j,:}=e_{\omega(q+\Delta)}.$ On the other hand, we decompose $V_{q}^{\mathrm{SA}}=W^{\mathrm{SA}}A_{q}$ with $(W^{\mathrm{SA}})_{:,h}=\operatorname{vect}(W^{(h)})$ and $(A_{\pmb q})_{h,i}=a_{\pmb q,\omega(i)}^{(h)}$  

The proof is concluded by showing that $\mathrm{row}(E_{q})\subseteq\mathrm{row}(A_{q})$ is a necessary and suffcient condition for the existence of a $W^{\mathrm{SA}}$ such that any $V_{q}^{\mathrm{conv}}=W^{\mathrm{conv}}E_{q}$ can be writen as $W^{\mathrm{SA}}A_{q}$  

Suffcient. Given that r $\operatorname{v}(E_{q})\subseteq\operatorname{row}(A_{q})$ there exists $\Phi\in\mathbb{R}^{K^{2}\times N_{h}}$ such that $E_{q}=\Phi A_{q}$ and a valid decomposition is $W^{\mathrm{SA}}=W^{\mathrm{conv}}\Phi$ which gives $W^{\mathrm{SA}}A_{q}=V_{q}^{\mathrm{conv}}$  

Necessary. Assume there exists $\pmb{x}\in\mathbb{R}^{H W}$ such that $\pmb{x}\in\mathrm{row}(\pmb{E}_{q})$ and $\pmb{x}\notin\operatorname{row}(\pmb{A}_{q})$ and set $\pmb{x}^{\top}$ to be a row of $V_{q}^{\mathrm{conv}}$ Then, $W^{\mathrm{SA}}A_{q}\ne V_{q}^{\mathrm{conv}}$ for any $W^{\mathrm{SA}}$ and there is no possibledecomposition.  

# E  GENERALIZED QUADRATIC POSITIONAL ENCODING  

We noticed the similarity of the attention probabilities in the quadratic positional encoding (Section 3) to isotropic bivariate Gaussian distributions with bounded support:  

$$
\mathrm{softmax}(\pmb{\mathsf{A}}_{q,:})_{k}=\frac{e^{-\alpha\|(\pmb{k}-\pmb{q})-\pmb{\Delta}\|^{2}}}{\sum_{\pmb{k}^{\prime}\in[W]\times[H]}e^{-\alpha\|(\pmb{k}^{\prime}-\pmb{q})-\pmb{\Delta}\|^{2}}}\,.
$$  

Building on this observation, we further extended our attention mechanism to non-isotropic Gaussian distribution over pixel positions. Each head is parametrized by a center of attention $\Delta$ and a covariancematrix $\Sigma$ to obtain the following attention scores,  

$$
\mathsf{A}_{q,k}=-\frac{1}{2}(\delta-\Delta)^{\top}\Sigma^{-1}(\delta-\Delta)=-\frac{1}{2}\delta^{\top}\Sigma^{-1}\delta+\delta^{\top}\Sigma^{-1}\Delta-\frac{1}{2}\Delta^{\top}\Sigma^{-1}\Delta\,,
$$  

where, once more, $\pmb{\delta}=\pmb{k}-\pmb{q}$ . The last term can be discarded because the softmax is shift invariant and we rewrite the attention coefficient as a dot product between the head target vector $\pmb{v}$ and the relative position encoding $r_{\delta}$ (consisting of the first and second order combinations of the shift in pixels $\delta$  

$$
v=\frac{1}{2}(2(\Sigma^{-1}\Delta)_{1},2(\Sigma^{-1}\Delta)_{2},-\Sigma_{1,1}^{-1},-\Sigma_{2,2}^{-1},-2\cdot\Sigma_{1,2}^{-1})^{\top}\mathrm{~and~}r_{\delta}=(\delta_{1},\delta_{2},\delta_{1}^{2},\delta_{2}^{2},\delta_{1}\delta_{2})^{\top}\,.
$$  

Evaluation.  We trained our model using this generalized quadratic relative position encoding. We were curious to see if, using the above encoding the self-attention model would learn to attend to non-isotropic groups of pixels—thus forming unseen patterns in CNNs. Each head was parametrized by $\pmb{\Delta}\in\mathbb{R}^{2}$ and $\bar{\Sigma}^{-1/2}\in\mathbb{R}^{2\times2}$ to ensure that the covariance matrix remained positive semi-definite. We initialized the center of attention to $\Delta^{(h)}\sim\mathcal{N}(\mathbf{0},2I_{2})$ and $\pmb{\Sigma}^{-1/2}=I_{2}+\mathcal{N}(\mathbf{0},0.01I_{2})$ so that initial attention probabilities were close to an isotropic Gaussian. Figure 12 shows that the network did learn non-isotropic attention probability patterns, especially in high layers. Nevertheless, the fact that we do not obtain any performance improvement seems to suggest that attention non-isotropy is not particularly helpful in practice—the quadratic positional encoding suffices.  

![](images/5889844ce70ca88cb449bb1755c438bc434592d7b077079f53bce03ee2c19e8b.jpg)  

Figure 12: Centers of attention of each attention head (different colors) for the 6 self-attention layers using non-isotropic Gaussian parametrization. The central black square is the query pixel, whereas solid and dotted circles represent the $50\%$ and $90\%$ percentiles of each Gaussian, respectively.  

Pruning degenerated heads. Some non-isotropic attention heads attend on “non-intuitive' patches of pixels: either attending a very thin stripe of pixels, when $\Sigma^{-1}$ was almost singular, or attending all pixels uniformly, when $\Sigma^{-1}$ was close to O (i.e. constant attention scores). We asked ourselves, are such attention patterns indeed useful for the model or are these heads degenerated and unused? To find out, we pruned all heads having largest eigen-values smaller than $10^{-{\overline{{5}}}}$ or condition number (ratio of the biggest and smallest eigen-values) greater than $10^{5}$ . Specifically in our model with 6-layer and 9-heads each, we pruned $[2,4,1,2,6,0]$ heads from the first to the last layer. This means that these layers cannot express a $3\times3$ kernel anymore. As shown in yellow on fig. 2, this ablation initially hurts a bit the performance, probably due to off biases, but after a few epochs of continued training with a smaller learning rate (divided by 10) the accuracy recovers its unpruned value. Hence, without sacrificing performance, we reduce the size of the parameters and the number of FLOPS by a fourth.  

# FINCREASING THE NUMBER OF HEADS  

For completeness, we also tested increasing the number of heads of our architecture from 9 to 16.  

![](images/dd479067b004a97e5759c54032b46ff542664dc9af2fc58fd2653e33f34194a4.jpg)  

Figure 13: Evolution of test accuracy on CIFAR- Table 4: Number of parameters and accuracy 10. Pruned model (yellow) is continued training on CIFAR-10 per model. SA stands for Selfof the non-isotropic model (orange). Attention.  

![](images/016eb4019403f9554a5bd7edb2f0e93e3f514a4b3eb76c3465861d41d85e1ae8.jpg)  

Figure 14: Centers of attention for 16 attention heads (different colors) for the 6 self-attention layers using quadratic positional encoding. The central black square is the query pixel, whereas solid and dotted circles represent the $50\%$ and $90\%$ percentiles of each Gaussian, respectively.  

Similar to Figure 4, we see that the network distinguishes two main types of attention patterns. Localized heads (i.e., those that attend to nearly individual pixels) appear more frequently in the first few layers. The self-attention layer uses these heads to act in a manner similar to how convolutional layers do. Heads with less-localized attention become more common at higher layers.  