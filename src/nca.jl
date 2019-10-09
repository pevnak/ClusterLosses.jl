
struct NCA{T<:Number}
	α::T
end

NCA() = NCA(1)

"""
function loss(l::NCA, d::AbstractMatrix, y)

	d --- a square matrix with distances
	y --- Vector with labels

"""
function loss(l::NCA, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	@assert size(d,1) == size(d,2) == length(y)
	idxs = labelmap(y)
	o, n = zero(T), 0
	for i in 1:size(d,2)
		yᵢ = y[i]
		os = sum(exp(-d[i,j]) for j in setdiff(idxs[yᵢ], i))
		oo = sum(exp(-d[i,j]) for j in setdiff(1:size(d,2), idxs[yᵢ]))
		o += -log(os) + log(oo)
	end
	o / size(d,2)
end

function ∇loss(Δ, l::NCA, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	@assert size(d,1) == size(d,2) == length(y)
	idxs = labelmap(y)
	o, n = zero(d), 0
	for i in 1:size(d,2)
		yᵢ = y[i]
		os = sum(exp(-d[i,j]) for j in setdiff(idxs[yᵢ], i))
		oo = sum(exp(-d[i,j]) for j in setdiff(1:size(d,2), idxs[yᵢ]))

		for j in setdiff(idxs[yᵢ], i)
			o[i,j] += Δ * exp(-d[i,j]) / os 
		end

		for j in setdiff(1:size(d,2), idxs[yᵢ])
			o[i,j] -= Δ * exp(-d[i,j]) / oo
		end
	end
	o / size(d,2)
end

Zygote.@adjoint function loss(l::NCA, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	return(loss(l, d, y), Δ -> (nothing, ∇loss(Δ, l, d, y), nothing))
end
