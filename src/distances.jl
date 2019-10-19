function _euclid(x)
	xn = sum(x.^2, dims = 1)
	transpose(xn) .- 2*transpose(x) * x .+ xn
end

function _euclid(x, y)
	xn = sum(x.^2, dims = 1)
	yn = sum(y.^2, dims = 1)
	transpose(xn) .- 2*transpose(x) * y .+ yn
end

function _cosine(x)
	xn = x ./ sqrt.(sum(x.^2, dims = 1) .+ eps(eltype(x)))
	1 .- transpose(xn) * xn
end

function _cosine(x, y)
	xn = x ./ sqrt.(sum(x.^2, dims = 1) .+ eps(eltype(x)))
	yn = y ./ sqrt.(sum(y.^2, dims = 1) .+ eps(eltype(y)))
	1 .- transpose(xn) * yn
end



function segmented_mean(x, bags) 
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j, b) in enumerate(bags)
        for bi in b
           for i in 1:size(x, 1)
                o[i, j] += x[i, bi]
			end
        end
        o[:, j] ./= length(b)
    end
    o
end

function ∇segmented_mean(Δ, x, bags) 
    dx = similar(x)
    for (j, b) in enumerate(bags)
        ws = view(Δ, :, j) ./ length(b)
        @inbounds for bi in b
            for i in 1:size(x, 1)
                dx[i, bi] = ws[i]
            end
        end
    end
    (dx, nothing)
end

Zygote.@adjoint function segmented_mean(x, bags::Vector{Vector{T}}) where {T<:Integer}
    return(segmented_mean(x, bags), Δ -> ∇segmented_mean(Δ, x, bags))
end

function _euclidmean(x, y::Vector{T}) where {T<:Integer}
	yc = labelmap(y)
	bags = [yc[k] for k in keys(yc)]
	segmented_mean(x, bags)
end

Zygote.@adjoint function _euclidmean(x, y) where {T}
	yc = labelmap(y)
	bags = [yc[k] for k in keys(yc)]
	return(segmented_mean(x, bags), Δ -> ∇segmented_mean(Δ, x, bags))
end
