
struct TripletLoss{T<:Number}
	α::T
end

TripletLoss() = TripletLoss(1)

"""
function loss(l::TripletLoss, d::AbstractMatrix, y)

	d --- a square matrix with distances
	y --- Vector with labels

"""
function loss(l::TripletLoss, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	@assert size(d,1) == size(d,2) == length(y)
	idxs = labelmap(y)
	o, n = zero(T), 0
	for i in 1:size(d,2)
		yᵢ = y[i]
		for j in setdiff(idxs[yᵢ], i)
			for k in setdiff(1:size(d,2), idxs[yᵢ])
				o += max(0, d[i,j] - d[i,k] + l.α)
				n += 1
			end
		end
	end
	o / n
end

function ∇loss(Δ, l::TripletLoss, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	@assert size(d,1) == size(d,2) == length(y)
	idxs = labelmap(y)
	o, n = zero(d), 0
	for i in 1:size(d,2)
		yᵢ = y[i]
		for j in setdiff(idxs[yᵢ], i)
			for k in setdiff(1:size(d,2), idxs[yᵢ])
				if (d[i,j] - d[i,k] + l.α) > 0
					o[i,j] += Δ
					o[i,k] -= Δ
				end
				n += 1
			end
		end
	end
	o ./ n
end

Zygote.@adjoint function loss(l::TripletLoss, d::AbstractMatrix{T}, y::AbstractVector) where {T}
	return(loss(l, d, y), Δ -> (nothing, ∇loss(Δ, l, d, y), nothing))
end
