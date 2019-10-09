# ClusterLosses.jl
Loss function to learn distance metrics

So far we have only triplet loss (*Weinberger,  Kilian Q. and Saul,  Lawrence K.   Distance metric learning for large margin nearestneighbor classification.J. Mach. Learn. Res., 10:207–244, June 2009. ISSN 1532-4435.*) and NCA loss (*Goldberger, Jacob, Roweis, Sam, Hinton, Geoff, and Salakhutdinov, Ruslan.  Neighbourhood com-ponents analysis.  InAdvances in Neural Information Processing Systems 17, pp. 513–520. MITPress, 2004.*). 

The losses works as follows
```
using ClusterLoss
y = [1,1,2,2]
d =  [0.67  0.25  1.46  0.63; 0.25  0.36  0.37  0.61; 1.46  0.37  0.77  1.64; 0.63  0.61  1.64  0.92]
loss(Triplet(1), d, y)
loss(NCA(), d, y)
```
where it is assumed that `d` is the distance matrix, e.g. Euclidean, Cosine, etc.

The loss functions are compatible with Flux, i.e. gradients are provided. 
