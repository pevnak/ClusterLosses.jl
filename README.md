# ClusterLosses.jl
Loss function to learn distance metrics

So far we have only triplet loss and NCA loss. It is as follows

```
	l = 
	y = [1,1,2,2]
	d =  [0.67  0.25  1.46  0.63;
	 	  0.25  0.36  0.37  0.61;
 	 	  1.46  0.37  0.77  1.64;
 	      0.63  0.61  1.64  0.92]
loss(Triplet(1), d, y)
loss(NCA(), d, y)
```
