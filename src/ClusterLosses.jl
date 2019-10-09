module ClusterLosses
using Zygote

include("utils.jl")
include("tripletloss.jl")
include("nca.jl")
export Triplet, NCA, loss
end # module
