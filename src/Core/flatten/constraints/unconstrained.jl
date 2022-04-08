############################################################################################
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter - keeps parameter unconstrained. Useful for assigning buffer values for functions of parameter.

# Fields
$(TYPEDFIELDS)
"""
struct Unconstrained <: AbstractConstraint
    function Unconstrained()
        return new()
    end
end

############################################################################################
function construct_transform(constrained::Unconstrained, val)
    return Bijectors.Identity{0}(), Bijectors.inverse(Bijectors.Identity{0}())
end

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::Unconstrained,
    val::Union{R,Array{R},AbstractArray}
) where {R<:Real}
    return true
end

############################################################################################
# Unconstrained constraint -> just flatten
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::Unconstrained,
    val
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    return construct_flatten(T, flattentype, unflattentype, val)
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
    Unconstrained,
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
