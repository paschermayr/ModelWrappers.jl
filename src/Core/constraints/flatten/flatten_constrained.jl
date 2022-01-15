############################################################################################
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter - keeps scalar parameter constrained.

# Fields
$(TYPEDFIELDS)
"""
struct Constrained{T<:Real} <: AbstractConstraint
    min::T
    max::T
    function Constrained(min::S, max::T, ϵ = 1e-6) where {S<:AbstractFloat, T<:AbstractFloat}
        ArgCheck.@argcheck min < max
        vals = promote(min, max)
        return new{eltype(vals)}(vals[1] + ϵ, vals[2] - ϵ)
    end
end
#=
#!NOTE: Bijectors seems to constrain Int to Float -> so we will remove this option
function Constrained(min::I, max::I) where {I<:Integer}
    ArgCheck.@argcheck min < max
    return Constrained{I}(min, max)
end
=#

############################################################################################
function _to_bijector(info::Constrained)
    return Bijectors.TruncatedBijector(info.min, info.max)
end

############################################################################################
function _checkparam(
    _rng::Random.AbstractRNG, val::R, constraint::Constrained
) where {R<:Real}
    ArgCheck.@argcheck typeof(val) == typeof(constraint.min) == typeof(constraint.max) "Type of Constrained boundaries must match type of val"
    ArgCheck.@argcheck constraint.min < val < constraint.max "val must be between boundaries"
    return true
end

############################################################################################
# Constrained constraint -> just flatten
function flatten(
    output::Type{T}, flattentype::F, unflattentype::U, x, constraint::Constrained
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    return flatten(T, flattentype, unflattentype, x)
end

############################################################################################
# Export
export Constrained, flatten
