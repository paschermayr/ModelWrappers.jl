############################################################################################
"""
$(TYPEDEF)

Objective struct with additional information about AD backend and configuration.

# Fields
$(TYPEDFIELDS)
"""
struct DiffObjective{O<:Objective,T<:AbstractDifferentiableTune}
    "Objective as function of a parameter vector in unconstrained space."
    objective::O
    "Automatic Differentiation configurations."
    tune::T
end

############################################################################################
function _log_density(diff::DiffObjective, θᵤ::AbstractVector{T}) where {T<:Real}
    return diff.objective(θᵤ)
end
function _log_density_and_gradient(
    diff::DiffObjective, θᵤ::AbstractVector{T}
) where {T<:Real}
    return _log_density_and_gradient(diff.objective, diff.tune, θᵤ)
end

############################################################################################
#export
export DiffObjective
