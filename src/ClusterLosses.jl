module ClusterLosses
using Zygote, Distances, LinearAlgebra

include("utils.jl")
include("tripletloss.jl")
include("nca.jl")
include("distances.jl")

loss(l, ::Distances.SqEuclidean, x, y) = loss(l, _euclid(x), y)
loss(l, ::Distances.CosineDist, x, y) = loss(l, _cosine(x), y)

export Triplet, NCA, loss
end # module
