# ClusterLosses.jl
Loss function to learn distance metrics

So far we have 
  * `Triplet(1)` Triplet loss  (*Weinberger,  Kilian Q. and Saul,  Lawrence K.   Distance metric learning for large margin nearestneighbor classification.J. Mach. Learn. Res., 10:207–244, June 2009. ISSN 1532-4435.*) 
  *`NCA()` Neighbourhood components analysis loss  (*Goldberger, Jacob, Roweis, Sam, Hinton, Geoff, and Salakhutdinov, Ruslan.  Neighbourhood components analysis.  In Advances in Neural Information Processing Systems 17, pp. 513–520. MITPress, 2004.*). 
  * `NCM()` Nearest Class Mean  *Mensink, Thomas, et al. "Distance-based image classification: Generalizing to new classes at near-zero cost." IEEE transactions on pattern analysis and machine intelligence 35.11 (2013): 2624-2637.*

The losses works as follows
```
using ClusterLosses
y = [1,1,2,2];
d =  [0.67  0.25  1.46  0.63; 0.25  0.36  0.37  0.61; 1.46  0.37  0.77  1.64; 0.63  0.61  1.64  0.92];
loss(Triplet(1), d, y)
loss(NCA(), d, y)
```
where it is assumed that `d` is the distance matrix, e.g. Euclidean, Cosine, etc.

If you want to calculate the distance from a matrix `x`m then do
```
using ClusterLosses, Distances
y = [1,1,2,2];
x = rand(2,4)
map(l -> loss(l, SqEuclidean(), x, y), [Triplet(1), NCA(), NCM()])
map(l -> loss(l, CosineDist(), x, y), [Triplet(1), NCA(), NCM()])
```
Note that at the moment, we support only SqEuclidean distance and Cosine Similarity


The loss functions are compatible with Flux, i.e. gradients are provided. 
```
using Flux, ClusterLosses, Distances
y = [1,1,2,2];
x = rand(2,4)
map(l -> gradient(x -> loss(l, CosineDist(), x, y), x)[1], [Triplet(1), NCA(), NCM()])
map(l -> gradient(x -> loss(l, SqEuclidean(), x, y), x)[1], [Triplet(1), NCA(), NCM()])
```
