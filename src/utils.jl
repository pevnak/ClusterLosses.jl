function labelmap(y::AbstractVector{T}) where {T}
	d = Dict{T, Vector{Int}}()
	for (i,v) in enumerate(y)
		if haskey(d, v)
			push!(d[v], i)
		else
			d[v] = [i]
		end
	end
	d
end

