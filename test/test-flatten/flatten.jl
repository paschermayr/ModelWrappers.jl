############################################################################################
function check_AD_closure(constraint, val)
    reconstruct = Reconstructor(constraint, val)
    bij, bij⁻¹ = construct_transform(constraint, val)
    function check_AD(θₜ::AbstractVector{T}) where {T<:Real}
        θ = unflattenAD(reconstruct, θₜ)
        θ = constrain(bij⁻¹, θ)
        return log_abs_det_jac(bij, θ)
    end
end

############################################################################################
@testset "Flatten - base" begin
    include("types.jl")
    include("nested.jl")
    include("constraints.jl")
end

############################################################################################
# Export
