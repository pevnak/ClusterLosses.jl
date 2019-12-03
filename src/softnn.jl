"""
	SoftNN(ϵ = 1f-6)
	
	
"""
struct SoftNN{T}
	τ::T
	ϵ::T
end

SoftNN() = SoftNN(1f0, 1f-6)
SoftNN(τ::T) where {T}= SoftNN(τ, T(1f-6))


loss(l::SoftNN, d, y) = loss(NCA(l.ϵ), d ./ l.τ, y)