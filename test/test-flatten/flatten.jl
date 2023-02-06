############################################################################################
function check_AD_closure(constraint, val)
    reconstruct = ReConstructor(constraint, val)
    transform = TransformConstructor(constraint, val)
    function check_AD(θₜ::AbstractVector{T}) where {T<:Real}
        θ_temp = unflattenAD(reconstruct, θₜ)
        θ = constrain(transform, θ_temp)
        return log_abs_det_jac(transform, θ)
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
