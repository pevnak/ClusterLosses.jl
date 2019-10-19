"""
	NCM(ϵ = 1f-6)
	
	Neighborhood components analysis loss function of  	*Mensink, Thomas, et al. "Distance-based image classification: Generalizing to new classes at near-zero cost." IEEE transactions on pattern analysis and machine intelligence 35.11 (2013): 2624-2637.*
"""
struct NCM{T}
	ϵ::T
end

NCM() = NCM(1f-6)


function _loss(l::NCM, d::AbstractMatrix{T}, clusters::Vector{Vector{I}}, iclusters::Vector{I}) where {T,I<:Integer}
	o, n, nc = zero(T), size(d, 1), size(d,2)
	for i in 1:n
		cluster = iclusters[i]
		os = d[i,cluster]
		oo = sum(exp(-d[i,c]) for c in setdiff(1:nc, cluster))
		o += os + log(oo + l.ϵ)
	end
	o / n
end

function _∇loss(Δ, l::NCM, d::AbstractMatrix{T}, clusters::Vector{Vector{I}}, iclusters::Vector{I}) where {T,I<:Integer}
	o, n, nc = zero(d), size(d, 1), size(d, 2)
	for i in 1:n
		cluster = iclusters[i]
		o[i,cluster] = Δ
		oo = sum(exp(-d[i,c]) for c in setdiff(1:nc, cluster))
		for c in setdiff(1:nc, cluster)
			o[i,c] -= Δ * exp(-d[i,c]) / (oo + l.ϵ)
		end
	end
	o / n
end

Zygote.@adjoint function _loss(l::NCM, d::AbstractMatrix{T}, clusters, iclusters) where {T}
	return(_loss(l, d, clusters, iclusters), Δ -> (nothing, _∇loss(Δ, l, d, clusters, iclusters), nothing, nothing))
end

function loss(l::NCM, ::Distances.CosineDist, x, y) 
	clusters, iclusters = identifyclusters(y)
	x = x ./ sqrt.(sum(x .^2, dims = 1) .+ eps(eltype(x)))
	μ = segmented_mean(x, clusters)
	d = _cosine(x, μ)
	_loss(l, d, clusters, iclusters)
end

function loss(l::NCM, ::Distances.SqEuclidean, x, y) 
	clusters, iclusters = identifyclusters(y)
	μ = segmented_mean(x, clusters)
	d = _euclid(x, μ)
	_loss(l, d, clusters, iclusters)
end