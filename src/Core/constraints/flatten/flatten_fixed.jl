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
function _to_bijector(info::Fixed)
    return Bijectors.Identity{0}()
end

############################################################################################
function _checkparam(
    _rng::Random.AbstractRNG, val::Union{R,Array{R},AbstractArray}, constraint::Fixed
) where {R<:Real}
    return true
end

############################################################################################
# Fixed constraint
function _flatten(
    output::Type{T}, flattentype::F, unflattentype::U, x
) where {T<:Real,F<:FlattenTypes,U<:UnflattenTypes}
    v = T[]
    _unflatten_Fixed(v) = x
    return v, _unflatten_Fixed
end
function flatten(
    output::Type{T}, flattentype::F, unflattentype::U, x, constraint::Fixed
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    return _flatten(T, flattentype, unflattentype, x)
end

############################################################################################
# Export
export Fixed, flatten
