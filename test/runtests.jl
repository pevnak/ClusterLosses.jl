using ClusterLosses, Test
using Distances, Statistics
using ClusterLosses: labelmap, loss, ∇loss, _euclidmean, segmented_mean
using FiniteDifferences, Flux

@testset "labelmap" begin 
	y = [1,2,3,2,1]
	idxs = labelmap(y)
	@test idxs[1] ≈ [1,5]
	@test idxs[2] ≈ [2,4]
	@test idxs[3] ≈ [3]
end


@testset "distances" begin
    x = randn(2,4);
    y = randn(2,4);
    @test pairwise(SqEuclidean(), x, dims = 2) ≈ ClusterLosses._euclid(x)
    @test pairwise(SqEuclidean(), x, y, dims = 2) ≈ ClusterLosses._euclid(x, y)
    @test pairwise(CosineDist(), x, dims = 2) ≈ ClusterLosses._cosine(x)
    @test pairwise(CosineDist(), x, y, dims = 2) ≈ ClusterLosses._cosine(x, y)
end

@testset "segmented_mean" begin 
	bags = [[3, 4], [5], [1, 2]]
	x = [0.22941573387056174 0.4588314677411235 0.22941573387056174 0.4588314677411235 0.6882472016116852]
	@test segmented_mean(x, bags) ≈  [0.3441236008058426 0.6882472016116852 0.3441236008058426]

	y = [1,1,2,2,3]
 	x = randn(2,5);
 	yc = labelmap(y)
	bags = [yc[k] for k in keys(yc)]
	fdm = central_fdm(5, 1);
	@test  _euclidmean(x, y) ≈	 reduce(hcat, [mean(x[:,b], dims = 2) for b in bags])
	@test gradient(x -> sum(sin.(_euclidmean(x, y))), x)[1] ≈ grad(fdm, x -> sum(sin.(_euclidmean(x, y))), x)
end

@testset "Tripletloss" begin 
	l = Triplet(1)
	y = [1,1,2,2]
	d =  [0.67  0.25  1.46  0.63;
	 	  0.25  0.36  0.37  0.61;
 	 	  1.46  0.37  0.77  1.64;
 	      0.63  0.61  1.64  0.92]
 	x = randn(2,4);

	o  = max(0, d[1,2] - d[1,3] + 1) + max(0, d[1,2] - d[1,4] + 1)
	o += max(0, d[1,2] - d[2,3] + 1) + max(0, d[2,1] - d[2,4] + 1)
	o += max(0, d[3,4] - d[3,1] + 1) + max(0, d[3,4] - d[3,2] + 1)
	o += max(0, d[3,4] - d[4,1] + 1) + max(0, d[3,4] - d[4,2] + 1)
	o /= 8
	@test loss(l, d, y) ≈ o


	fdm = central_fdm(5, 1);
	@test grad(fdm, d -> sum(loss(l, d, y)), d) ≈ ClusterLosses.∇loss(1, l, d, y)
	@test gradient(d -> sum(sin.(loss(l, d, y))), d)[1] ≈ grad(fdm, d -> sum(sin.(loss(l, d, y))), d)
	@test gradient(x -> loss(l, SqEuclidean() , x, y), x)[1] ≈ grad(fdm, x -> sum(loss(l, SqEuclidean() , x, y)), x)
	@test gradient(x -> loss(l, CosineDist() , x, y), x)[1] ≈ grad(fdm, x -> sum(loss(l, CosineDist() , x, y)), x)

	d =  [0.0   0.0  1.5  1.5;
	 	  0.0   0.0  1.5  1.5;
 	 	  1.5   1.5	 0.0  0.0;
 	      1.5   1.5	 0.0  0.0]
 	@test loss(l, d, y) == 0
	d =  [1.5   1.5	 0.0  0.0;
		  1.5   1.5	 0.0  0.0;
		  0.0   0.0  1.5  1.5;
	 	  0.0   0.0  1.5  1.5]
 	@test loss(l, d, y) == 2.5
end

@testset "NCA loss" begin 
	l = NCA(0)
	y = [1,1,2,2]
	d =  [0.67  0.25  1.46  0.63;
	 	  0.25  0.36  0.37  0.61;
 	 	  1.46  0.37  0.77  1.64;
 	      0.63  0.61  1.64  0.92]
 	x = randn(2,4);

	o   = -log(exp(-d[1,2])) + log(exp(-d[1,3]) + exp(-d[1,4]))
	o  += -log(exp(-d[1,2])) + log(exp(-d[2,3]) + exp(-d[2,4]))
	o  += -log(exp(-d[3,4])) + log(exp(-d[1,3]) + exp(-d[2,3]))
	o  += -log(exp(-d[3,4])) + log(exp(-d[1,4]) + exp(-d[2,4]))
	@test loss(l, d, y) ≈ o / 4

	fdm = central_fdm(5, 1);
	@test grad(fdm, d -> sum(loss(l, d, y)), d) ≈ ClusterLosses.∇loss(1, l, d, y)
	@test gradient(d -> sum(sin.(loss(l, d, y))), d)[1] ≈ grad(fdm, d -> sum(sin.(loss(l, d, y))), d)
	@test gradient(x -> loss(l, SqEuclidean() , x, y), x)[1] ≈ grad(fdm, x -> sum(loss(l, SqEuclidean() , x, y)), x)
	@test gradient(x -> loss(l, CosineDist() , x, y), x)[1] ≈ grad(fdm, x -> sum(loss(l, CosineDist() , x, y)), x)
	d =  [0.0   0.0  1.5  1.5;
	 	  0.0   0.0  1.5  1.5;
 	 	  1.5   1.5	 0.0  0.0;
 	      1.5   1.5	 0.0  0.0]
 	@test loss(l, d, y) == -0.8068528194400547
	d =  [1.5   1.5	 0.0  0.0;
		  1.5   1.5	 0.0  0.0;
		  0.0   0.0  1.5  1.5;
	 	  0.0   0.0  1.5  1.5]
 	@test loss(l, d, y) == 2.1931471805599454
end



@testset "NCM loss" begin 
	l = NCM(0)
	y = [1,1,2,2,3]
 	x = randn(2,5);
	fdm = central_fdm(5, 1);

 	x₁ = [1.0 1 2 2 3];
 	x₂ = [1.0 2 1 2 3];
 	@test loss(l, SqEuclidean(), x₁, y) < loss(l, SqEuclidean(), x₂, y)
 	x₁ = [1.0 1 0 0 -1;
 		  0   0 1 1  0];
 	x₂ = [1.0 0 1 0 -1;
 		  0   1 0 1  0];
 	@test loss(l, CosineDist(), x₁, y) < loss(l, CosineDist(), x₂, y)
end

