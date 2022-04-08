############################################################################################
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter - keeps parameter fixed. Useful for assigning buffer values for functions of parameter.

# Fields
$(TYPEDFIELDS)
"""
struct Fixed <: AbstractConstraint
    function Fixed()
        return new()
    end
end

############################################################################################
function construct_transform(info::Fixed, val)
    return Bijectors.Identity{0}(), Bijectors.Identity{0}()
end

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::Fixed,
    val::Union{R,Array{R},AbstractArray},
) where {R<:Real}
    return true
end

############################################################################################
# Fixed constraint
function _construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    x
) where {T<:Real,F<:FlattenTypes,U<:UnflattenTypes}
    v = T[]
    _flatten_Fixed(x) = v
    _unflatten_Fixed(v) = x
    return _flatten_Fixed, _unflatten_Fixed
end
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::Fixed,
    x,
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    return _construct_flatten(T, flattentype, unflattentype, x)
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
    Fixed,
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
