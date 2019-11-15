
struct Triplet{T<:Number}
	α::T
end

Triplet() = Triplet(1f0)

"""
function loss(l::Triplet, d::AbstractMatrix, y)

	d --- a square matrix with distances
	y --- Vector with labels

"""
function loss(l::Triplet, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	@assert size(d,1) == size(d,2) == length(y)
	idxs = labelmap(y)
	o₁, o₂, n₁, n₂ = zero(T),zero(T), 0, 0
	for i in 1:size(d,2)
		yᵢ = y[i]
		for j in setdiff(idxs[yᵢ], i)
			for k in setdiff(1:size(d,2), idxs[yᵢ])
				o₁ += max(0, d[i,j] - d[i,k] + l.α)
				n₁ += 1
			end
			o₂ += d[i,j]
			n₂ += 1
		end
	end
	o₁ / n₁ + o₂ / n₂
end

function ∇loss(Δ, l::Triplet, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	@assert size(d,1) == size(d,2) == length(y)
	idxs = labelmap(y)
	o₁, n₁ = zero(d), 0
	o₂, n₂ = zero(d), 0
	for i in 1:size(d,2)
		yᵢ = y[i]
		for j in setdiff(idxs[yᵢ], i)
			for k in setdiff(1:size(d,2), idxs[yᵢ])
				if (d[i,j] - d[i,k] + l.α) > 0
					o₁[i,j] += Δ
					o₁[i,k] -= Δ
				end
				n₁ += 1
			end
			o₂[i,j] += Δ
			n₂ += 1
		end
	end
	o₁ / n₁ + o₂ / n₂
end

Zygote.@adjoint function loss(l::Triplet, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	return(loss(l, d, y), Δ -> (nothing, ∇loss(Δ, l, d, y), nothing))
end
