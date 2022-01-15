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
function _to_bijector(info::Unconstrained)
    return Bijectors.Identity{0}()
end

############################################################################################
function _checkparam(
    _rng::Random.AbstractRNG, val::Union{R,Array{R},AbstractArray}, constraint::Unconstrained
) where {R<:Real}
    return true
end

############################################################################################
# Unconstrained constraint -> just flatten
function flatten(
    output::Type{T}, flattentype::F, unflattentype::U, x, constraint::Unconstrained
) where {T<:AbstractFloat,F<:FlattenTypes,U<:UnflattenTypes}
    return flatten(T, flattentype, unflattentype, x)
end

############################################################################################
# Export
export Unconstrained, flatten
