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
function construct_transform(info::Constrained, val)
    transform = Bijectors.TruncatedBijector(info.min, info.max)
    return transform, Bijectors.inverse(transform)
end

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::Constrained,
    val::R,
) where {R<:Real}
    ArgCheck.@argcheck typeof(val) == typeof(constraint.min) == typeof(constraint.max) "Type of Constrained boundaries must match type of val"
    ArgCheck.@argcheck constraint.min < val < constraint.max "val must be between boundaries"
    return true
end

############################################################################################
# Constrained constraint -> just flatten
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::Constrained,
    x
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    return construct_flatten(T, flattentype, unflattentype, x)
end

############################################################################################
#=
!NOTE: For this constraint, we use a Bijector as Transformer, so we do not need to add a new functors
1. MyTransformer <: AbstractTransformer, MyInverseTransformer <: AbstractTransformer
2. define a function construct_transform(MyConstraint, val) -> MyTransformer, MyInverseTransformer
3. overload unconstrain and log_abs_det_jac on MyTransformer.
4. overload constrain on MyInverseTransformer.
=#

############################################################################################
#Export
export
    Constrained,
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
