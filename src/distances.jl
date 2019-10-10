# definition of gradients of pairwise
function _euclid(x)
	xn = sum(x.^2, dims = 1)
	transpose(xn) .- 2*transpose(x) * x .+ xn
end

function _cosine(x)
	xn = x ./ sqrt.(sum(x.^2, dims = 1) .+ eps(eltype(x)))
	1 .- transpose(xn) * xn
end
