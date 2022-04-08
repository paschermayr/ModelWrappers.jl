############################################################################################
"""
$(TYPEDEF)

Utility struct to help assign boundaries to parameter.

# Fields
$(TYPEDFIELDS)
"""
struct Simplex <: AbstractConstraint
    len::Int64
    function Simplex(len::Integer)
        ArgCheck.@argcheck len > 0
        return new(len)
    end
end
Simplex(vec::AbstractVector) = Simplex(length(vec))

############################################################################################
function construct_transform(constraint::Simplex, val)
    transform = Bijectors.SimplexBijector{1, true}()
    return transform, Bijectors.inverse(transform)
end

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    constraint::Simplex,
    val::Vector{R},
) where {R<:Real}
    ArgCheck.@argcheck length(val) == constraint.len
    ArgCheck.@argcheck all(val[iter] > 0.0 for iter in eachindex(val))
    return true
end

############################################################################################
#= !NOTES:
    (1) Constrained's last element will sum up to 1.
    (2) Unconstrained's last element is irrelevant for Bijector{Simplex} in default method.
    (3) Consequently, we can flatten in length(x)-1 dimensions, and unflatten back to length(x) by summing up elements for length(x)'s element.
    This works for both constrained/unconstrained.
=#
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    constraint::C,
    x::Vector{R},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    R<:Real,
    C<:Union{Simplex, Distributions.Dirichlet, Bijectors.SimplexBijector}
}
    buffer_flat = zeros(T, length(x)-1)
    len_flat = length(x)
    len_unflat = len_flat-1
    function flatten_Simplex(x_vec::AbstractVector{R}) where {R<:Real}
        ArgCheck.@argcheck length(x_vec) == len_flat
        return fill_array!(buffer_flat, view(x_vec, 1:len_flat-1))
    end
    buffer_unflat = zeros(R, length(x))
    function unflatten_Simplex(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len_unflat
        return Simplex_from_flatten!(buffer_unflat, v)
    end
    return flatten_Simplex, unflatten_Simplex
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    constraint::C,
    x::Vector{R},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    R<:Real,
    C<:Union{Simplex, Distributions.Dirichlet, Bijectors.SimplexBijector}
}
    len_flat = length(x)
    len_unflat = len_flat-1
    function flatten_Simplex_AD(x_vec::AbstractVector{R}) where {R<:Real}
        ArgCheck.@argcheck length(x_vec) == len_flat
        buffer = zeros(R, len_flat-1)
        return fill_array!(buffer, view(x_vec, 1:len_flat-1))
    end
    function unflatten_Simplex_AD(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len_unflat
        buffer = zeros(eltype(v), length(x))
        return Simplex_from_flatten!(buffer, v)
    end
    return flatten_Simplex_AD, unflatten_Simplex_AD
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
    Simplex,
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
