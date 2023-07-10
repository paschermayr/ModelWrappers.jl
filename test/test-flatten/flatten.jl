############################################################################################
function check_AD_closure(constraint, val)
    reconstruct = ReConstructor(constraint, val)
    transform = TransformConstructor(constraint, val)
    info = ParameterInfo(FlattenDefault(), reconstruct, transform, val)
    function check_AD(θₜ::AbstractVector{T}) where {T<:Real}
        θ = unflattenAD_constrain(info, θₜ)
        return log_abs_det_jac(transform, θ)
    end
end

############################################################################################
@testset "Flatten - base" begin
    include("types.jl")
    include("constraints.jl")
    include("nested.jl")
end

############################################################################################
# Export
