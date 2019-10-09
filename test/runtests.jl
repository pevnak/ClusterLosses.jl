using ClusterLosses, Test
using ClusterLosses: labelmap, loss, ∇loss
using FiniteDifferences, Flux

@testset "labelmap" begin 
	y = [1,2,3,2,1]
	idxs = labelmap(y)
	@test idxs[1] ≈ [1,5]
	@test idxs[2] ≈ [2,4]
	@test idxs[3] ≈ [3]
end

@testset "tripletloss" begin 
	l = TripletLoss(1)
	y = [1,1,2,2]
	d =  [0.67  0.25  1.46  0.63;
	 	  0.25  0.36  0.37  0.61;
 	 	  1.46  0.37  0.77  1.64;
 	      0.63  0.61  1.64  0.92]

	o  = max(0, d[1,2] - d[1,3] + 1) + max(0, d[1,2] - d[1,4] + 1)
	o += max(0, d[1,2] - d[2,3] + 1) + max(0, d[2,1] - d[2,4] + 1)
	o += max(0, d[3,4] - d[3,1] + 1) + max(0, d[3,4] - d[3,2] + 1)
	o += max(0, d[3,4] - d[4,1] + 1) + max(0, d[3,4] - d[4,2] + 1)
	o /= 8

	@test loss(l, d, y) ≈ o


	fdm = central_fdm(5, 1);
	@test grad(fdm, d -> sum(loss(l, d, y)), d) ≈ ClusterLosses.∇loss(1, l, d, y)
	@test gradient(d -> sum(sin.(loss(l, d, y))), d)[1] ≈ grad(fdm, d -> sum(sin.(loss(l, d, y))), d)
end

