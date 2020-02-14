"""
	labelmap(y)

	dictionary with labels to indexes
"""
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

Zygote.@nograd labelmap

"""
	clusters, iclusters = identifyclusters(y)

	create a vector of indexes of samples with the same label
	and its inverse mapping
"""
function identifyclusters(y)
	yc = labelmap(y)
	clusters = [yc[k] for k in keys(yc)]
	iclusters = Vector{Int}(undef, length(y))
	for (i,cluster) in enumerate(clusters)
		for j in cluster
			iclusters[j] = i 
		end
	end
	return(clusters, iclusters)
end

Zygote.@nograd identifyclusters